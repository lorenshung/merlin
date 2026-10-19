"""Structural native-unit eligibility, independent of source overload names.

Representation splitting and exact host readout do not change a producer's
mathematical role. Parallel runtime products, nonlinear rounding, and predicate
identities must not become native units merely because component lowering can
express them with contractions. Index arithmetic remains address selection.
"""


def construction_evidence(module):
    from merlin.kernels.shapes import observe_contractions

    evidence = []
    contractions = bool(observe_contractions(module)) or any(
        op.name in {"scf.for", "linalg.reduce"}
        or (op.name == "linalg.generic" and any(it.data.value == "reduction" for it in op.get_iterator_types()))
        for op in module.walk()
    )

    def literal(value, seen=None):
        seen = set() if seen is None else set(seen)
        if value in seen:
            return False
        seen.add(value)
        owner = value.owner
        name = getattr(owner, "name", None)
        if name in {"arith.constant", "linalg.index"}:
            return True
        if name == "linalg.generic":
            block = owner.body.block
            return literal(block.last_op.operands[0], seen)
        if name in {
            "tensor.splat",
            "arith.extf",
            "arith.truncf",
            "arith.sitofp",
            "arith.uitofp",
            "arith.extsi",
            "arith.extui",
            "arith.trunci",
            "tensor.reshape",
            "tensor.cast",
        }:
            return literal(owner.operands[0], seen)
        if name is not None and (name.startswith("arith.") or name.startswith("math.")):
            return bool(owner.operands) and all(literal(v, seen) for v in owner.operands)
        if name is None and hasattr(owner, "parent_op"):
            parent = owner.parent_op()
            if getattr(parent, "name", None) == "linalg.generic" and value.index < len(parent.inputs):
                return literal(parent.inputs[value.index], seen)
        return False

    for ordinal, op in enumerate(module.walk()):
        if op.name in {"math.floor", "math.ceil", "math.trunc", "math.round", "math.roundeven"}:
            if not literal(op.operands[0]):
                evidence.append(dict(ordinal=ordinal, operation=op.name, reason="nonlinear_rounding_construction"))
        boolean_product = op.name == "arith.andi" and str(op.results[0].type) == "i1"
        if op.name not in {"arith.mulf", "arith.muli"} and not boolean_product:
            continue
        if any(str(value.type) == "index" for value in op.operands) or any(literal(v) for v in op.operands):
            continue
        parent = op.parent_op()
        while parent is not None and parent.name != "linalg.generic":
            parent = parent.parent_op()
        if parent is None or any(it.data.value != "parallel" for it in parent.get_iterator_types()):
            continue
        # A contraction epilogue can multiply runtime scale/seed components.
        # It is not the standalone Hadamard-to-diagonal construction.
        if not contractions:
            evidence.append(dict(ordinal=ordinal, operation=op.name, reason="parallel_runtime_product_construction"))
    return dict(schema="merlin.native_unit_policy.v1", status="host" if evidence else "eligible", evidence=evidence)
