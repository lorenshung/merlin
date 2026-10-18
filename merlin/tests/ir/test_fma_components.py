"""Fused floating products use integer MACs and one rational result boundary."""

import pytest
from test_ieee_component_epilogues import scalar_bits

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.floating_components import rewrite_floating_components
from merlin.llvmlower.integer_components import rewrite_integer_components
from merlin.targetgen.core_aten_components import unlowered_data_reductions


def source(token):
    return parse_mlir_text(f"""module {{
      func.func @forward(%x:tensor<1x{token}>,%y:tensor<1x{token}>,%z:tensor<1x{token}>) -> tensor<1x{token}> {{
        %e=tensor.empty() : tensor<1x{token}>
        %r=linalg.generic {{indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,
          affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}}
          ins(%x,%y,%z:tensor<1x{token}>,tensor<1x{token}>,tensor<1x{token}>) outs(%e:tensor<1x{token}>) {{
          ^bb0(%a:{token},%b:{token},%c:{token},%out:{token}):
            %v=math.fma %a,%b,%c : {token}
            linalg.yield %v : {token}
        }} -> tensor<1x{token}>
        return %r : tensor<1x{token}>
      }} }}""")


@pytest.mark.parametrize("token,width,bias", [("f32", 32, 127), ("f64", 64, 1023)])
def test_fma_zero_sign_and_exact_cancellation(token, width, bias):
    module = source(token)
    _, proofs = rewrite_floating_components(module, [(0, 1)] * 3)
    assert len(proofs) == 1 and proofs[0]["source_data_arithmetic"]
    final = [op for op in module.walk() if op.name == "linalg.generic"][-1]
    sign = 1 << (width - 1)
    for a in [0, sign]:
        for b in [0, sign]:
            for c in [0, sign]:
                expected = sign if bool(a ^ b) and bool(c) else 0
                assert scalar_bits(final.body.block, [0, 0, a, b, c, 0]) == expected
    fraction = 23 if width == 32 else 52
    one = bias << fraction
    assert scalar_bits(final.body.block, [0, 0, one, one, one | sign, 0]) == 0


@pytest.mark.parametrize("grids", [[(-23, 2**23 + 1), (-23, 2**23 - 1), (0, 1)], [(-100, 1), (20, 1), (-120, 1)]])
def test_fma_has_full_integer_producers_and_one_final_rounding(grids):
    module = source("f32")
    bounds, proofs = rewrite_floating_components(module, grids)
    assert proofs and proofs[0]["rounding"] == "integer_quotient_remainder_rne"
    assert not any(op.name == "math.fma" for op in module.walk())
    plans = rewrite_integer_components(module, limb_attribute="test.limb", input_bounds=[0] * 3, known_bounds=bounds)
    assert plans
    assert not unlowered_data_reductions(module, "test.limb")
    module.verify()


@pytest.mark.parametrize("token", ["f16", "bf16"])
@pytest.mark.parametrize("coefficient", [0, 257])
def test_reduced_storage_scalar_coefficients_keep_their_finite_grid(token, coefficient):
    from merlin.llvmlower.integer_float_boundaries import rewrite_integer_float_boundaries

    module = parse_mlir_text(f"""module {{
      func.func @forward(%x:tensor<1x{token}>) -> tensor<1x{token}> {{
        %one=arith.constant 1.0 : f32
        %raw=arith.truncf %one : f32 to {token}
        %nanword=arith.constant 32767 : i16
        %nan=arith.bitcast %nanword : i16 to {token}
        %missing=arith.cmpf uno,%one,%one : f32
        %literal=arith.select %missing,%nan,%raw : {token}
        %other=tensor.splat %literal : tensor<1x{token}>
        %e=tensor.empty() : tensor<1x{token}>
        %r=linalg.generic {{indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,
          affine_map<(d0)->(d0)>],iterator_types=["parallel"]}}
          ins(%x,%other:tensor<1x{token}>,tensor<1x{token}>) outs(%e:tensor<1x{token}>) {{
          ^bb0(%a:{token},%b:{token},%unused:{token}):
            %integer=arith.constant {coefficient} : i64
            %fp=arith.sitofp %integer : i64 to f32
            %alpha=arith.truncf %fp : f32 to {token}
            %aa=arith.extf %a : {token} to f32
            %bb=arith.extf %b : {token} to f32
            %cc=arith.extf %alpha : {token} to f32
            %v=math.fma %bb,%cc,%aa : f32
            %out=arith.truncf %v : f32 to {token}
            linalg.yield %out : {token}
        }} -> tensor<1x{token}>
        return %r : tensor<1x{token}>
      }} }}""")
    grids = {}
    _, count = rewrite_integer_float_boundaries(module, [None], output_grids=grids)
    assert count == 1 and list(grids.values()) == [(0, 2 * coefficient)]
    _, proofs = rewrite_floating_components(module, [(0, 1)], known_grids=grids)
    assert any(p["source_data_arithmetic"] for p in proofs)
    assert not any(o.name == "math.fma" for o in module.walk())
    module.verify()
