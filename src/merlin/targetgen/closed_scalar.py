"""Exact evaluation of scalar IR on closed inputs, and closed value intervals.

Used by source specializations that must take their meaning from the IR
itself: a literal exponent, a literal divisor, a mask compared against a
literal over a provably bounded value. Every unsupported operation, runtime
leaf or non-finite intermediate is UNKNOWN, so callers fail closed.
"""

import math
from fractions import Fraction

UNKNOWN = object()
FLOAT_BITS = {"f16": (16, False), "bf16": (16, True), "f32": (32, False), "f64": (64, False)}


def _name(op):
    """Operation name, including unregistered operations parsed from generic syntax."""
    from merlin.targetgen.native_payload_carriers import op_name

    return op_name(op)


def _element_type(value):
    typ = value.type
    return getattr(typ, "element_type", typ)


def literal(value):
    """A closed uniform literal behind generic block arguments and splats, else None."""
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr

    seen = set()
    while value not in seen:
        seen.add(value)
        owner = value.owner
        name = _name(owner)
        if name == "arith.constant":
            attr = owner.properties.get("value", owner.attributes.get("value"))
            if hasattr(attr, "value") and hasattr(attr.value, "data"):
                data = attr.value.data
                return bool(data) if str(_element_type(value)) == "i1" and isinstance(data, int) else data
            if isinstance(attr, DenseIntOrFPElementsAttr):
                values = list(attr.get_values())
                if values and all(str(v) == str(values[0]) for v in values):
                    return values[0]
            return None
        if name == "tensor.splat":
            value = owner.operands[0]
            continue
        if name is None and hasattr(owner, "parent_op"):
            parent = owner.parent_op()
            if getattr(parent, "name", None) == "linalg.generic" and value.index < len(parent.inputs):
                value = parent.inputs[value.index]
                continue
        return None
    return None


def round_to(value, typ):
    """One IEEE RNE of a real value to a floating type; integers wrap to their width."""
    from xdsl.dialects.builtin import IntegerType

    from merlin.targetgen.exact_arithmetic import round_binary

    if isinstance(typ, IntegerType):
        width = typ.width.data
        integer = int(value) % (1 << width)
        if width == 1:
            return bool(integer)
        return integer - (1 << width) if integer >= 1 << (width - 1) else integer
    bits = FLOAT_BITS.get(str(typ))
    if bits is None:
        return UNKNOWN
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value) or value == 0.0):
        return value
    if value == 0:
        return 0.0
    try:
        return round_binary(Fraction(value), bits[0], bfloat16=bits[1], overflow="infinity")
    except (ValueError, OverflowError):
        return UNKNOWN


def _finite(*values):
    return all(isinstance(v, (int, float, bool)) and not (isinstance(v, float) and not math.isfinite(v)) for v in values)


def _zero_sign(value):
    return math.copysign(1.0, value) < 0 if isinstance(value, float) else False


def evaluate(value, environment, memo=None):
    """Evaluate one scalar SSA value. environment maps values to Python numbers."""
    from xdsl.dialects.builtin import IntegerType

    from merlin.llvmlower.singleton_selectors import _relation

    memo = {} if memo is None else memo
    if value in environment:
        return environment[value]
    if value in memo:
        return memo[value]
    known = literal(value)
    if known is not None:
        memo[value] = known
        return known
    owner = value.owner
    name = _name(owner)
    result_type = _element_type(value)

    def get(v):
        return evaluate(v, environment, memo)

    out = UNKNOWN
    if name in {"arith.sitofp", "arith.uitofp"}:
        x = get(owner.operands[0])
        if x is not UNKNOWN:
            source = _element_type(owner.operands[0])
            integer = int(x)
            if name == "arith.uitofp" and isinstance(source, IntegerType):
                integer %= 1 << source.width.data
            out = round_to(integer, result_type)
    elif name in {"arith.extui", "arith.extsi", "arith.trunci"}:
        x = get(owner.operands[0])
        if x is not UNKNOWN:
            source = _element_type(owner.operands[0])
            integer = int(x)
            if name == "arith.extui" and isinstance(source, IntegerType):
                integer %= 1 << source.width.data
            out = round_to(integer, result_type)
    elif name in {"arith.extf", "arith.truncf"}:
        x = get(owner.operands[0])
        if x is not UNKNOWN:
            out = x if name == "arith.extf" else round_to(x, result_type)
    elif name == "math.absf":
        x = get(owner.operands[0])
        if x is not UNKNOWN and isinstance(x, float):
            out = abs(x)
    elif name == "arith.negf":
        x = get(owner.operands[0])
        if x is not UNKNOWN and isinstance(x, float):
            out = -x
    elif name in {"arith.mulf", "arith.addf", "arith.subf", "arith.divf"}:
        a, b = get(owner.operands[0]), get(owner.operands[1])
        if a is not UNKNOWN and b is not UNKNOWN and _finite(a, b):
            a, b = float(a), float(b)
            if name == "arith.mulf":
                exact = Fraction(a) * Fraction(b)
                sign = _zero_sign(a) != _zero_sign(b) if a == 0 or b == 0 else exact < 0
            elif name in {"arith.addf", "arith.subf"}:
                b = -b if name == "arith.subf" else b
                exact = Fraction(a) + Fraction(b)
                sign = _zero_sign(a) and _zero_sign(b)
            else:
                if b == 0:
                    exact = None
                else:
                    exact = Fraction(a) / Fraction(b)
                    sign = _zero_sign(a) != _zero_sign(b)
            if exact is not None:
                rounded = round_to(exact, result_type)
                if rounded is not UNKNOWN and rounded == 0:
                    rounded = -0.0 if sign else 0.0
                out = rounded
    elif name == "arith.remf":
        a, b = get(owner.operands[0]), get(owner.operands[1])
        if a is not UNKNOWN and b is not UNKNOWN and _finite(a, b) and float(b) != 0:
            out = math.fmod(float(a), float(b))  # IEEE fmod is exact; zero takes the dividend's sign.
    elif name in {"math.powf", "math.fpowi", "math.ipowi"}:
        base, exponent = get(owner.operands[0]), get(owner.operands[1])
        if exponent is not UNKNOWN and exponent == 0:
            out = round_to(1, result_type)
        elif base is not UNKNOWN and exponent is not UNKNOWN and _finite(base, exponent):
            if exponent > 0 and float(base) in (0.0, 1.0) and math.copysign(1.0, float(base)) > 0:
                out = round_to(float(base), result_type)  # 0^e = +0 and 1^e = 1 for every e > 0.
            elif float(exponent).is_integer() and exponent > 0:
                out = round_to(Fraction(base) ** int(exponent), result_type)
                if out == 0 and _zero_sign(float(base)) and int(exponent) % 2:
                    out = -0.0
    elif name in {"arith.cmpf", "arith.cmpi"}:
        a, b = get(owner.operands[0]), get(owner.operands[1])
        if a is not UNKNOWN and b is not UNKNOWN:
            predicate = owner.predicate.value.data
            if name == "arith.cmpf":
                unordered = (isinstance(a, float) and math.isnan(a)) or (isinstance(b, float) and math.isnan(b))
                relation = 0 if unordered else (a > b) - (a < b)
                out = _relation(predicate, relation, unordered, floating=True)
            else:
                typ = _element_type(owner.operands[0])
                if isinstance(typ, IntegerType):
                    width = typ.width.data
                    ua, ub = int(a) % (1 << width), int(b) % (1 << width)
                    if predicate in {2, 3, 4, 5}:
                        ua = ua - (1 << width) if ua >= 1 << (width - 1) else ua
                        ub = ub - (1 << width) if ub >= 1 << (width - 1) else ub
                    out = _relation(predicate, (ua > ub) - (ua < ub))
    elif name in {"arith.andi", "arith.ori", "arith.xori"}:
        a, b = get(owner.operands[0]), get(owner.operands[1])
        boolean = str(result_type) == "i1"
        if boolean and name == "arith.andi" and (a is False or b is False):
            out = False
        elif boolean and name == "arith.ori" and (a is True or b is True):
            out = True
        elif a is not UNKNOWN and b is not UNKNOWN:
            x, y = int(a), int(b)
            out = round_to(x & y if name == "arith.andi" else x | y if name == "arith.ori" else x ^ y, result_type)
    elif name == "arith.select":
        condition = get(owner.operands[0])
        if condition is not UNKNOWN:
            out = get(owner.operands[1] if condition else owner.operands[2])
    memo[value] = out
    return out


class Interval:
    """A closed real interval of the values a tensor or scalar can hold; None bounds mean empty."""

    def __init__(self, lo=None, hi=None):
        self.lo, self.hi = lo, hi

    @property
    def empty(self):
        return self.lo is None

    def union(self, other):
        if other is None:
            return None
        if self.empty:
            return other
        if other.empty:
            return self
        return Interval(min(self.lo, other.lo), max(self.hi, other.hi))

    def __eq__(self, other):
        return isinstance(other, Interval) and (self.lo, self.hi) == (other.lo, other.hi)


def interval(value, memo=None, active=None):
    """Closed interval of every element value can take, or None when unbounded/unknown.

    tensor.empty contributes no value: its contents are unspecified until written,
    so a read of an unwritten element may be refined to any written value.
    """
    from xdsl.dialects.builtin import IntegerType

    memo = {} if memo is None else memo
    active = set() if active is None else active
    if value in memo:
        return memo[value]
    if value in active:
        return Interval()  # Loop-carried: the fixpoint below re-checks the assumption.
    known = literal(value)
    if known is not None and _finite(known):
        return Interval(Fraction(known), Fraction(known))
    owner = value.owner
    name = _name(owner)
    active.add(value)

    def get(v):
        return interval(v, memo, active)

    out = None
    try:
        if name == "tensor.empty":
            out = Interval()
        elif name == "arith.andi":
            typ = _element_type(value)
            for mask_operand in owner.operands:
                mask = literal(mask_operand)
                if isinstance(typ, IntegerType) and isinstance(mask, int) and 0 <= mask < 1 << (typ.width.data - 1):
                    out = Interval(Fraction(0), Fraction(mask))
                    break
        elif name in {"arith.uitofp", "arith.sitofp"}:
            source = get(owner.operands[0])
            if source is not None and (name == "arith.sitofp" or source.empty or source.lo >= 0):
                if source.empty:
                    out = source
                else:
                    lo, hi = round_to(source.lo, _element_type(value)), round_to(source.hi, _element_type(value))
                    if _finite(lo, hi):
                        out = Interval(Fraction(lo), Fraction(hi))
        elif name in {"arith.mulf", "arith.addf"}:
            for index in (0, 1):
                scale = literal(owner.operands[1 - index])
                source = get(owner.operands[index]) if scale is not None and _finite(scale) else None
                if source is None:
                    continue
                if source.empty:
                    out = source
                    break
                ends = (
                    [source.lo * Fraction(scale), source.hi * Fraction(scale)]
                    if name == "arith.mulf"
                    else [source.lo + Fraction(scale), source.hi + Fraction(scale)]
                )
                rounded = [round_to(v, _element_type(value)) for v in ends]
                if _finite(*rounded):
                    out = Interval(Fraction(min(rounded)), Fraction(max(rounded)))
                break
        elif name == "tensor.insert":
            scalar, destination = get(owner.operands[0]), get(owner.operands[1])
            out = None if scalar is None or destination is None else scalar.union(destination)
        elif name == "tensor.extract":
            out = get(owner.operands[0])
        elif name == "scf.for":
            index = value.index
            block = owner.body.block
            initial = get(owner.iter_args[index])
            carried = block.args[index + 1]
            out = initial
            for _ in range(8):
                if out is None:
                    break
                memo[carried] = out
                local = dict(memo)
                yielded = interval(block.last_op.operands[index], local, set())
                widened = None if yielded is None else out.union(yielded)
                if widened == out:
                    break
                out = widened
            else:
                out = None
            memo.pop(carried, None)
        elif name == "scf.if":
            out = Interval()
            for region in owner.regions:
                branch = get(region.block.last_op.operands[value.index])
                out = None if branch is None or out is None else out.union(branch)
        elif name is None and hasattr(owner, "parent_op"):
            parent = owner.parent_op()
            if getattr(parent, "name", None) == "linalg.generic" and value.index < len(parent.inputs):
                out = get(parent.inputs[value.index])
        elif name == "linalg.generic":
            out = get(owner.body.block.last_op.operands[value.index])
    finally:
        active.discard(value)
    memo[value] = out
    return out
