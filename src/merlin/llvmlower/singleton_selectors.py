"""Prove closed compare/select regions to be storage selectors or fills.

One runtime scalar is partitioned into complete typed comparison cells. Each
predicate is constant throughout a cell; the cells cover every storage word,
including float NaNs and both zero signs. No tensor values are consulted.
Unknown arithmetic, multiple runtime inputs and non-singleton reductions refuse.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Literal:
    value: object
    typ: object

    def key(self):
        from xdsl.dialects.builtin import IntegerType

        if isinstance(self.typ, IntegerType):
            return str(self.typ), int(self.value) % (1 << self.typ.width.data)
        if str(self.typ) == "index":
            return "index", int(self.value)
        return str(self.typ), self.typ.pack([self.value])


RUNTIME = object()
IGNORED = object()


def _relation(predicate, relation, unordered=False, *, floating=False):
    eq, gt, lt = relation == 0, relation > 0, relation < 0
    if floating:
        ordered = not unordered
        return (
            False,
            ordered and eq,
            ordered and gt,
            ordered and not lt,
            ordered and lt,
            ordered and not gt,
            ordered and not eq,
            ordered,
            unordered or eq,
            unordered or gt,
            unordered or not lt,
            unordered or lt,
            unordered or not gt,
            unordered or not eq,
            unordered,
            True,
        )[predicate]
    return (eq, not eq, lt, not gt, gt, not lt, lt, not gt, gt, not lt)[predicate]


def _evaluate(op, environment, cell, thresholds, element):
    from xdsl.dialects.builtin import IntegerType, i1

    values = [environment[v] for v in op.operands]
    name = op.name
    typ = op.results[0].type
    if name == "arith.select":
        if not isinstance(values[0], Literal):
            raise ValueError("unproved selector predicate")
        return values[1 if values[0].value else 2]
    if name in {"arith.andi", "arith.ori", "arith.xori"} and str(typ) == "i1":
        if not all(isinstance(v, Literal) for v in values):
            raise ValueError("runtime Boolean arithmetic")
        a, b = [bool(v.value) for v in values]
        return Literal({"arith.andi": a and b, "arith.ori": a or b, "arith.xori": a != b}[name], typ)
    if name in {"arith.index_cast", "arith.index_castui"} and isinstance(values[0], Literal):
        return Literal(int(values[0].value), typ)
    if name not in {"arith.cmpi", "arith.cmpf"}:
        raise ValueError("unproved scalar operation")
    predicate = op.predicate.value.data
    if name == "arith.cmpf":

        def position(value):
            if value is RUNTIME:
                return cell
            if math.isnan(value.value):
                return None
            return 2 * thresholds.index(value.value)

        a, b = map(position, values)
        unordered = a is None or b is None
        relation = 0 if unordered else (a > b) - (a < b)
        return Literal(_relation(predicate, relation, unordered, floating=True), i1)
    signed = predicate in {2, 3, 4, 5}
    operand_type = op.operands[0].type
    width = operand_type.width.data if isinstance(operand_type, IntegerType) else None

    def interval(value):
        if value is RUNTIME:
            lo, hi = cell
        else:
            lo = hi = int(value.value)
            if width is not None:
                lo = hi = lo % (1 << width)
        if signed and width is not None and lo >= 1 << (width - 1):
            lo -= 1 << width
            hi -= 1 << width
        return lo, hi

    a, b = map(interval, values)
    if values[0] is RUNTIME and values[1] is RUNTIME:
        return Literal(_relation(predicate, 0), i1)
    answers = {_relation(predicate, (x > y) - (x < y)) for x in a for y in b}
    if len(answers) != 1:
        raise ValueError("comparison cell straddles a predicate boundary")
    return Literal(answers.pop(), i1)


def _prove(op, ranges):
    from xdsl.dialects.builtin import IntegerType

    from merlin.xdsl_dialects.lowering.group_numerics import _constant_of

    block = op.body.block
    element = op.inputs[0].type.element_type
    integer = isinstance(element, IntegerType)
    if not integer and str(element) not in {"f16", "bf16", "f32", "f64"}:
        return None
    initial = {block.args[0]: RUNTIME}
    for arg, value in zip(list(block.args)[1:], op.outputs, strict=True):
        literal = _constant_of(value)
        if literal is None:
            if arg.uses:
                return None
            initial[arg] = IGNORED
        else:
            initial[arg] = Literal(literal, arg.type)
    for child in block.ops:
        for flag in ("fastmath", "overflowFlags"):
            value = child.properties.get(flag)
            if value is not None and str(value).split("<")[-1] != "none>":
                return None
        if child.name == "arith.constant":
            value = child.value
            if not hasattr(value, "value"):
                return None
            initial[child.results[0]] = Literal(value.value.data, child.results[0].type)
        elif child.name == "linalg.index":
            dimension = child.dim.value.data
            if ranges[dimension] != 1:
                return None
            initial[child.results[0]] = Literal(0, child.results[0].type)
    literals = [v.value for v in initial.values() if isinstance(v, Literal)]
    if integer:
        width = element.width.data
        limit = 1 << width
        boundaries = {0, limit, limit // 2}
        for value in literals:
            if isinstance(value, (int, bool)):
                word = int(value) % limit
                boundaries.update((word, word + 1))
        ordered = sorted(boundaries)
        cells = [(a, b - 1) for a, b in zip(ordered, ordered[1:]) if a < b]
        thresholds = []
    else:
        thresholds = sorted({-math.inf, math.inf, *[v for v in literals if not math.isnan(v)]})
        cells = [*range(2 * len(thresholds) - 1), None]
    results = []
    for cell in cells:
        environment = dict(initial)
        for child in block.ops:
            if child.name == "linalg.yield":
                results.append([environment[v] for v in child.operands])
                break
            if child.results and child.results[0] in initial:
                continue
            if len(child.results) != 1 or child.regions:
                return None
            environment[child.results[0]] = _evaluate(child, environment, cell, thresholds, element)
    if not results:
        return None
    proved = []
    for result, column in zip(op.results, zip(*results, strict=True), strict=True):
        if not result.uses:
            proved.append(IGNORED)
            continue
        first = next((value for value in column if isinstance(value, Literal)), None)

        def equals_literal(value, cell):
            if isinstance(value, Literal):
                return value.key() == first.key()
            return integer and value is RUNTIME and cell[0] == cell[1] == int(first.value) % (1 << element.width.data)

        if all(value is RUNTIME for value in column):
            proved.append(RUNTIME)
        elif isinstance(first, Literal) and all(
            equals_literal(value, cell) for value, cell in zip(column, cells, strict=True)
        ):
            proved.append(first)
        else:
            return None
    return proved


def fold_singleton_selectors(module):
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, FloatAttr, IntegerAttr, IntegerType, TensorType, i64
    from xdsl.rewriter import Rewriter

    from merlin.targetgen.native_payload_carriers import UNKNOWN, LiteralPayloads

    payloads = LiteralPayloads()
    count = 0
    for op in list(module.walk()):
        if op.name != "linalg.generic" or len(op.inputs) != 1 or not op.results:
            continue
        if len(op.outputs) != len(op.results):
            continue
        if payloads.classify(op.inputs[0]) & UNKNOWN:
            continue
        reductions = [i for i, it in enumerate(op.get_iterator_types()) if it.data.value == "reduction"]
        if not reductions and not any(child.name in {"arith.cmpi", "arith.cmpf"} for child in op.body.block.ops):
            continue
        try:
            ranges = op.get_static_loop_ranges()
            if any(n < 0 for n in ranges) or any(ranges[i] != 1 for i in reductions):
                continue
            proved = _prove(op, ranges)
        except (KeyError, IndexError, TypeError, ValueError, AttributeError, OverflowError):
            continue
        if proved is None or len(proved) != len(op.results):
            continue
        from math import prod

        from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr

        valid = True
        maps = list(op.get_indexing_maps())
        nontrivial = {i for i, n in enumerate(ranges) if n != 1}

        def project(amap, shape):
            positions = []
            if len(amap.results) != len(shape):
                return None
            for expr, extent in zip(amap.results, shape, strict=True):
                if isinstance(expr, AffineDimExpr) and ranges[expr.position] == extent:
                    if expr.position in nontrivial:
                        positions.append(expr.position)
                elif not (isinstance(expr, AffineConstantExpr) and expr.value == 0 and extent == 1):
                    return None
            return positions if len(positions) == len(set(positions)) and set(positions) == nontrivial else None

        for i, proof in enumerate(proved):
            if proof is IGNORED:
                continue
            target = project(maps[len(op.inputs) + i].data, op.results[i].type.get_shape())
            if target is None:
                valid = False
                break
            if proof is not RUNTIME:
                continue
            source = project(maps[0].data, op.inputs[0].type.get_shape())
            target = project(maps[len(op.inputs) + i].data, op.results[i].type.get_shape())
            if (
                source is None
                or source != target
                or op.results[i].type.element_type != op.inputs[0].type.element_type
                or prod(op.inputs[0].type.get_shape()) != prod(op.results[i].type.get_shape())
            ):
                valid = False
                break
        if not valid:
            continue
        operations, values = [], []
        for result, proof, seed in zip(op.results, proved, op.outputs, strict=True):
            if proof is IGNORED:
                values.append(seed)
            elif proof is RUNTIME:
                shape = list(result.type.get_shape())
                sizes = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(TensorType(i64, [len(shape)]), shape))
                value = tensor.ReshapeOp(op.inputs[0], sizes.result, result.type)
                operations.extend([sizes, value])
                values.append(value.result)
            else:
                typ = result.type.element_type
                attr = (
                    IntegerAttr(int(proof.value), typ) if isinstance(typ, IntegerType) else FloatAttr(proof.value, typ)
                )
                constant = arith.ConstantOp(attr)
                value = tensor.SplatOp(constant.result, [], result.type)
                operations.extend([constant, value])
                values.append(value.result)
        Rewriter.replace_op(op, operations, values)
        count += 1
    return count
