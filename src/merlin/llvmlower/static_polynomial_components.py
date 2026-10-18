"""Retain static gathers and express closed polynomial data work as integer MACs."""

from math import prod

from merlin.llvmlower.static_polynomials import PolynomialRefusal, analyze_static_polynomial


def lift_static_polynomial(op, grid, pack, bounds, *, operand_type=None, result_type=None):
    """Build a width-proved exact contraction, or refuse before changing source."""
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, IndexType, IntegerAttr, TensorType, i64, i128
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineDimExpr, AffineMap

    i64, i128 = operand_type or i64, result_type or i128
    operand_limit = i64.width.data - 2
    reconstruction_limit = i128.width.data - 8
    atoms, rows = analyze_static_polynomial(op)
    sources = list(dict.fromkeys(source for source, _ in atoms))
    grids = {source: grid(source) for source in sources}
    if any(value is None for value in grids.values()):
        raise PolynomialRefusal("polynomial operand lacks a finite packing proof")
    terms = []
    exponent = 0
    for row in rows:
        result = []
        for key, coefficient in row.items():
            e = -(coefficient.denominator.bit_length() - 1)
            if coefficient.denominator != 1 << -e:
                raise PolynomialRefusal("non-dyadic coefficient")
            numerator = coefficient.numerator
            e += sum(grids[atoms[index][0]][0] for index in key)
            result.append((key, numerator, e))
            exponent = min(exponent, e)
        terms.append(result)
    bound = max(
        (
            sum(abs(n) * (1 << (e - exponent)) * prod(grids[atoms[i][0]][1] for i in key) for key, n, e in row)
            for row in terms
        ),
        default=0,
    )
    if bound.bit_length() >= reconstruction_limit:
        raise PolynomialRefusal("polynomial reconstruction exceeds declared signed-width proof")
    operations, packed = [], {}
    for source in sources:
        value, _ = pack(source, grids[source][0], operations)
        if value.type.element_type != i64:
            raise PolynomialRefusal("polynomial gather operand width differs from its packing proof")
        packed[source] = value
    index_constants = {}

    def index(value):
        if value not in index_constants:
            constant = arith.ConstantOp(IntegerAttr(value, IndexType()))
            operations.append(constant)
            index_constants[value] = constant.result
        return index_constants[value]

    def integer(value):
        constant = arith.ConstantOp(IntegerAttr(value, i64))
        operations.append(constant)
        return constant.result

    atom_values = {}
    for i, (source, coordinates) in enumerate(atoms):
        extract = tensor.ExtractOp(packed[source], [index(v) for v in coordinates], i64)
        operations.append(extract)
        atom_values[i] = extract.result
    count = max(map(len, terms), default=0)
    if not count:
        raise PolynomialRefusal("empty polynomial")
    left_type, right_type = TensorType(i64, [len(rows), 1, count]), TensorType(i64, [len(rows), count, 1])
    left_empty, right_empty = tensor.EmptyOp([], left_type), tensor.EmptyOp([], right_type)
    operations.extend([left_empty, right_empty])
    left, right = left_empty.tensor, right_empty.tensor
    left_bound, right_bound = 0, 0
    for row_index, row in enumerate(terms):
        for term_index in range(count):
            key, coefficient, e = row[term_index] if term_index < len(row) else ((), 0, exponent)
            weight = coefficient * (1 << (e - exponent))
            a = atom_values[key[0]] if key else integer(1)
            b = atom_values[key[1]] if len(key) == 2 else integer(1)
            ab = grids[atoms[key[0]][0]][1] if key else 1
            bb = grids[atoms[key[1]][0]][1] if len(key) == 2 else 1
            if (abs(weight) * bb).bit_length() > operand_limit or ab.bit_length() > operand_limit:
                raise PolynomialRefusal("polynomial gather coefficient exceeds declared signed-width proof")
            product = arith.MuliOp(b, integer(weight))
            operations.append(product)
            li = tensor.InsertOp(a, left, [index(row_index), index(0), index(term_index)])
            ri = tensor.InsertOp(product.result, right, [index(row_index), index(term_index), index(0)])
            operations.extend([li, ri])
            left, right = li.result, ri.result
            left_bound, right_bound = max(left_bound, ab), max(right_bound, abs(weight) * bb)
    bounds[left], bounds[right] = left_bound, right_bound
    zero = arith.ConstantOp(IntegerAttr(0, i128))
    seed = tensor.SplatOp(zero.result, [], TensorType(i128, [len(rows), 1, 1]))
    operations.extend([zero, seed])
    bounds[seed.result] = 0
    block = Block(arg_types=[i64, i64, i128])
    a, b = arith.ExtSIOp(block.args[0], i128), arith.ExtSIOp(block.args[1], i128)
    product = arith.MuliOp(a.result, b.result)
    addition = arith.AddiOp(product.result, block.args[2])
    block.add_ops([a, b, product, addition, L.YieldOp(addition.result)])
    contraction = L.GenericOp(
        inputs=[left, right],
        outputs=[seed.result],
        body=Region(block),
        indexing_maps=ArrayAttr(
            [
                AffineMapAttr(AffineMap(4, 0, tuple(AffineDimExpr(d) for d in axes)))
                for axes in [(0, 1, 3), (0, 3, 2), (0, 1, 2)]
            ]
        ),
        iterator_types=ArrayAttr(
            [L.IteratorTypeAttr(L.IteratorType.PARALLEL)] * 3 + [L.IteratorTypeAttr(L.IteratorType.REDUCTION)]
        ),
        result_types=[seed.result.type],
    )
    operations.append(contraction)
    bounds[contraction.results[0]] = bound
    return operations, contraction.results[0], exponent, bound
