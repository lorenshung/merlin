# ruff: noqa: E501 -- MLIR fixtures retain one operation per line
"""Exact bit checks for signed-zero and separable invalid-result epilogues."""

import struct

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.fixed_point_ir import FORMAT
from merlin.llvmlower.floating_components import rewrite_floating_components
from merlin.llvmlower.ieee_zero_division import rewrite_zero_division
from merlin.targetgen.core_aten_components import unlowered_data_reductions


def scalar_bits(block, arguments):
    values = dict(zip(block.args, arguments, strict=True))

    def floating(bits, token):
        if token == "bf16":
            return struct.unpack("<f", (bits << 16).to_bytes(4, "little"))[0]
        return struct.unpack(
            {"f16": "<e", "f32": "<f", "f64": "<d"}[token],
            bits.to_bytes({"f16": 2, "f32": 4, "f64": 8}[token], "little"),
        )[0]

    for op in block.ops:
        args = [values[v] for v in op.operands]
        name = op.name
        if name == "linalg.yield":
            return args[0]
        if name == "arith.constant":
            value = op.value.value.data
            typ = str(op.results[0].type)
            if typ in FORMAT:
                value = int.from_bytes(op.results[0].type.pack([value]), "little")
        elif name == "arith.bitcast":
            value = args[0]
        elif name == "arith.select":
            value = args[1] if args[0] else args[2]
        elif name == "arith.cmpf":
            assert op.predicate.value.data == 1  # ordered equal
            value = int(floating(args[0], str(op.operands[0].type)) == floating(args[1], str(op.operands[1].type)))
        elif name == "arith.cmpi":
            predicate = op.predicate.value.data
            a, b = args
            value = int({0: a == b, 1: a != b, 6: a < b, 8: a > b}[predicate])
        elif name in {"arith.andi", "arith.ori", "arith.xori"}:
            a, b = args
            value = {"arith.andi": lambda: a & b, "arith.ori": lambda: a | b, "arith.xori": lambda: a ^ b}[name]()
        else:
            raise AssertionError(name)
        typ = op.results[0].type
        if hasattr(typ, "width"):
            value &= (1 << typ.width.data) - 1
        values[op.results[0]] = value
    raise AssertionError("missing yield")


def module(token, operation, divisor=None):
    literal = (
        f"%c=arith.constant {divisor} : {token}\n%constant=tensor.splat %c : tensor<1x{token}>"
        if divisor is not None
        else ""
    )
    rhs = "%constant" if divisor is not None else "%y"
    return parse_mlir_text(f"""module {{
      func.func @forward(%x: tensor<1x{token}>, %y: tensor<1x{token}>) -> tensor<1x{token}> {{
        {literal}
        %e=tensor.empty() : tensor<1x{token}>
        %r=linalg.generic {{indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}}
          ins(%x,{rhs} : tensor<1x{token}>,tensor<1x{token}>) outs(%e : tensor<1x{token}>) {{
        ^bb0(%a:{token},%b:{token},%unused:{token}):
          %v=arith.{operation} %a,%b : {token}
          linalg.yield %v : {token}
        }} -> tensor<1x{token}>
        return %r : tensor<1x{token}>
      }} }}""")


@pytest.mark.parametrize("token", ["f16", "bf16", "f32", "f64"])
@pytest.mark.parametrize("operation", ["addf", "subf"])
def test_exact_zero_uses_ieee_rne_operand_signs(token, operation):
    source = module(token, operation)
    _, proofs = rewrite_floating_components(source, [(0, 1), (0, 1)])
    assert proofs
    assert unlowered_data_reductions(source, "test.limb") == []
    final = list(op for op in source.walk() if op.name == "linalg.generic")[-1]
    f, e, bias = FORMAT[token]
    sign = 1 << (f + e)
    for left in (0, sign):
        for right in (0, sign):
            expected_negative = bool(left) and (not bool(right) if operation == "subf" else bool(right))
            assert scalar_bits(final.body.block, [0, 0, left, right, 0]) == (sign if expected_negative else 0)
    # Nonzero exact cancellation rounds to positive zero under RNE.
    one = bias << f
    right = one if operation == "subf" else one | sign
    assert scalar_bits(final.body.block, [0, 0, one, right, 0]) == 0


@pytest.mark.parametrize("token", ["f16", "bf16", "f32", "f64"])
@pytest.mark.parametrize("divisor", ["0.0", "-0.0"])
def test_zero_divisor_overlay_depends_on_the_computed_numerator(token, divisor):
    source = module(token, "divf", divisor)
    assert rewrite_zero_division(source) == 1
    producers = list(op for op in source.walk() if op.name == "linalg.generic")
    assert "merlin.exact_host_arithmetic" in producers[-1].attributes
    assert producers[-1].inputs[0].owner is producers[-2]
    f, e, bias = FORMAT[token]
    sign, inf = 1 << (f + e), ((1 << e) - 1) << f
    zero_sign = sign if divisor == "-0.0" else 0
    invalid = inf | 1 << (f - 1) | (0 if token == "bf16" else sign)
    for numerator in (0, sign):
        assert scalar_bits(producers[-1].body.block, [numerator, zero_sign, 0]) == invalid
    for numerator in (bias << f, (bias << f) | sign):
        assert scalar_bits(producers[-1].body.block, [numerator, zero_sign, 0]) == inf | (
            (numerator ^ zero_sign) & sign
        )


@pytest.mark.parametrize("token", ["f16", "bf16", "f32", "f64"])
def test_aggregate_propagation_keeps_special_words_and_discards_finite_terms(token):
    from merlin.llvmlower.ieee_aggregate import propagation_source

    source = module(token, "addf")
    op = next(op for op in source.walk() if op.name == "linalg.generic")
    propagation, proof = propagation_source(op, lambda value: (0, 2))
    assert proof["finite_dependence_outputs"] == "retain_exact_device_integer_readout_without_overlay"
    assert not any(op.name in {"arith.addf", "arith.mulf"} for op in propagation.walk())
    f, e, bias = FORMAT[token]
    inf, sign = ((1 << e) - 1) << f, 1 << (f + e)
    one, nan = bias << f, inf | 1 << (f - 1)
    invalid = nan | (0 if token == "bf16" else sign)
    assert scalar_bits(propagation.body.block, [one, one, 0]) == 0
    assert scalar_bits(propagation.body.block, [inf, one, 0]) == inf
    assert scalar_bits(propagation.body.block, [inf, inf | sign, 0]) == invalid
    assert scalar_bits(propagation.body.block, [nan, one, 0]) == nan


def test_aggregate_finite_overflow_requires_host_instruction_order():
    from merlin.llvmlower.ieee_aggregate import propagation_source

    source = module("f32", "addf")
    op = next(op for op in source.walk() if op.name == "linalg.generic")
    with pytest.raises(ValueError, match="host instruction order"):
        propagation_source(op, lambda value: (127, (1 << 24) - 1))


def test_input_special_overlay_cannot_cover_finite_intermediate_overflow():
    source = module("f32", "addf")
    with pytest.raises(ValueError, match="finite dependence can overflow"):
        rewrite_floating_components(source, [(127, 1), (127, 1)], ieee_pointwise=True)
