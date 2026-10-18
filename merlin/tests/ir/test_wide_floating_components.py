# ruff: noqa: E501 -- static MLIR fixtures retain one operation per line
"""Finite IEEE exponent spans use exact bit-field packing, never float scaling."""

from fractions import Fraction

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.fixed_point_ir import finite_grid
from merlin.llvmlower.floating_components import rewrite_floating_components
from merlin.llvmlower.integer_components import rewrite_integer_components


def test_extreme_bfloat_payload_has_full_integer_producers():
    values = [2.0**-133, -(2.0**127), 0.0]
    exponent, bound = finite_grid(values)
    assert exponent == -133 and bound == 1 << 260
    for value in values:
        assert Fraction.from_float(value) / Fraction(2) ** exponent == int(
            Fraction.from_float(value) / Fraction(2) ** exponent
        )
    module = parse_mlir_text("""module {
      func.func @forward(%x: tensor<3xbf16>) -> tensor<3xbf16> {
        %e = tensor.empty() : tensor<3xbf16>
        %r = linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>], iterator_types=["parallel"]}
          ins(%x : tensor<3xbf16>) outs(%e : tensor<3xbf16>) {
        ^bb0(%a:bf16,%b:bf16):
          %v=arith.addf %a,%a : bf16
          linalg.yield %v : bf16
        } -> tensor<3xbf16>
        return %r : tensor<3xbf16>
      }}""")
    bounds, proofs = rewrite_floating_components(module, [(exponent, bound)])
    assert proofs[0]["packing"] == "exact_ieee_bit_fields"
    assert proofs[0]["reconstruction"] == "bounded_i1024"
    assert not any(op.name in {"arith.fptosi", "arith.mulf"} for op in module.walk())
    assert all(
        op.operands[0].type.width.data <= 128 for op in module.walk() if op.name in {"arith.divui", "arith.remui"}
    )
    plans = rewrite_integer_components(module, limb_attribute="test.limb", known_bounds=bounds, input_bounds=[0])
    assert plans and max(entry["index"] for plan in plans for entry in plan["inputs"]) > 6
    module.verify()


def test_exact_widening_keeps_extreme_finite_grids_for_following_arithmetic():
    values = [2.0**-133, -(2.0**127), 0.0]
    module = parse_mlir_text("""module {
      func.func @forward(%x: tensor<3xbf16>) -> tensor<3xf32> {
        %e=tensor.empty() : tensor<3xf32>
        %a=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
          ins(%x:tensor<3xbf16>) outs(%e:tensor<3xf32>) {
        ^bb0(%v:bf16,%out:f32):
          %w=arith.extf %v : bf16 to f32
          linalg.yield %w : f32
        } -> tensor<3xf32>
        %r=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
          ins(%a:tensor<3xf32>) outs(%e:tensor<3xf32>) {
        ^bb1(%v:f32,%out:f32):
          %w=arith.addf %v,%v : f32
          linalg.yield %w : f32
        } -> tensor<3xf32>
        return %r : tensor<3xf32>
      }}""")
    bounds, proofs = rewrite_floating_components(module, [finite_grid(values)])
    assert any(proof["source_data_arithmetic"] for proof in proofs)
    assert not any(op.name == "arith.addf" for op in module.walk())
    plans = rewrite_integer_components(module, limb_attribute="test.limb", input_bounds=[0], known_bounds=bounds)
    assert plans
    module.verify()


def test_integer_promotion_grids_participate_in_the_packing_width_proof():
    module = parse_mlir_text("""module {
      func.func @forward(%x: tensor<5xf32>) -> tensor<5xf32> {
        %two=arith.constant -2.0 : f32
        %y=tensor.splat %two : tensor<5xf32>
        %e=tensor.empty() : tensor<5xf32>
        %r=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
          ins(%x,%y:tensor<5xf32>,tensor<5xf32>) outs(%e:tensor<5xf32>) {
        ^bb0(%a:f32,%b:f32,%out:f32):
          %w=arith.divf %a,%b : f32
          linalg.yield %w : f32
        } -> tensor<5xf32>
        return %r : tensor<5xf32>
      }}""")
    function = next(op for op in module.walk() if op.name == "func.func")
    # A preceding exact native integer->f32 boundary supplies this integral grid.
    bounds, proofs = rewrite_floating_components(
        module, [None], known_grids={function.body.block.args[0]: (0, 1 << 64)}
    )
    assert proofs and proofs[0]["source_data_arithmetic"]
    assert proofs[0]["packing"] == "exact_ieee_bit_fields"
    assert proofs[0]["reconstruction"] == "bounded_i1024"
    assert rewrite_integer_components(module, limb_attribute="test.limb", input_bounds=[0], known_bounds=bounds)
    module.verify()


def test_alignment_between_individually_short_grids_selects_wide_packing():
    module = parse_mlir_text("""module {
      func.func @forward(%x:tensor<1xf32>,%y:tensor<1xf32>) -> tensor<1xf32> {
        %e=tensor.empty() : tensor<1xf32>
        %r=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
          ins(%x,%y:tensor<1xf32>,tensor<1xf32>) outs(%e:tensor<1xf32>) {
        ^bb0(%a:f32,%b:f32,%out:f32):
          %w=arith.addf %a,%b : f32
          linalg.yield %w : f32
        } -> tensor<1xf32>
        return %r : tensor<1xf32>
      }}""")
    bounds, proofs = rewrite_floating_components(module, [finite_grid([2.0**-120]), finite_grid([2.0**100])])
    assert len(proofs) == 1 and proofs[0]["packing"] == "exact_ieee_bit_fields"
    assert not any(op.name == "arith.addf" for op in module.walk())
    assert rewrite_integer_components(module, limb_attribute="test.limb", input_bounds=[0, 0], known_bounds=bounds)
    module.verify()


def test_odd_division_retains_a_rounded_dyadic_grid_through_a_reshape():
    module = parse_mlir_text("""module {
      func.func @forward(%x:tensor<2xbf16>) -> tensor<2xbf16> {
        %three=arith.constant 3.0 : bf16
        %divisor=tensor.splat %three : tensor<2xbf16>
        %e=tensor.empty() : tensor<2xbf16>
        %v=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
          ins(%x,%divisor:tensor<2xbf16>,tensor<2xbf16>) outs(%e:tensor<2xbf16>) {
        ^bb0(%a:bf16,%b:bf16,%out:bf16):
          %w=arith.divf %a,%b : bf16
          linalg.yield %w : bf16
        } -> tensor<2xbf16>
        %shape=arith.constant dense<[2]> : tensor<1xi64>
        %reshaped=tensor.reshape %v(%shape) : (tensor<2xbf16>,tensor<1xi64>) -> tensor<2xbf16>
        %r=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
          ins(%reshaped:tensor<2xbf16>) outs(%e:tensor<2xbf16>) {
        ^bb1(%a:bf16,%out:bf16):
          %w=arith.addf %a,%a : bf16
          linalg.yield %w : bf16
        } -> tensor<2xbf16>
        return %r : tensor<2xbf16>
      }}""")
    bounds, proofs = rewrite_floating_components(module, [finite_grid([1.0, 2.0])])
    assert len(proofs) == 2 and all(p["source_data_arithmetic"] for p in proofs)
    assert not any(op.name in {"arith.addf", "arith.divf"} for op in module.walk())
    assert rewrite_integer_components(module, limb_attribute="test.limb", input_bounds=[0], known_bounds=bounds)
    module.verify()


def test_ieee_bit_epilogue_does_not_hide_the_complete_arithmetic_producer():
    from merlin.targetgen.core_aten_components import unlowered_data_reductions

    module = parse_mlir_text("""module {
      func.func @forward(%x:tensor<2xf32>,%y:tensor<2xf32>) -> tensor<2xf32> {
        %e=tensor.empty() : tensor<2xf32>
        %r=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
          ins(%x,%y:tensor<2xf32>,tensor<2xf32>) outs(%e:tensor<2xf32>) {
        ^bb0(%a:f32,%b:f32,%out:f32):
          %sum=arith.addf %a,%b : f32
          %bits=arith.bitcast %sum : f32 to i32
          %mask=arith.constant 2147483647 : i32
          %magnitude=arith.andi %bits,%mask : i32
          %v=arith.bitcast %magnitude : i32 to f32
          linalg.yield %v : f32
        } -> tensor<2xf32>
        return %r : tensor<2xf32>
      }}""")
    bounds, proofs = rewrite_floating_components(module, [(0, 2), (0, 2)])
    assert any(p["source_data_arithmetic"] for p in proofs)
    assert rewrite_integer_components(module, limb_attribute="test.limb", input_bounds=[0, 0], known_bounds=bounds)
    assert not unlowered_data_reductions(module, "test.limb")
    module.verify()
