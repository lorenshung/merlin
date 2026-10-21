"""Shape-only singleton monoids preserve complete integer words, not float NaNs."""

import pytest
from mlir_fixtures import assert_unchanged

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.structural_folds import fold_singleton_reductions


def source(element, arithmetic, seed, *, extent=1, flags=""):
    return parse_mlir_text(f"""module {{
      func.func @forward(%x: tensor<2x{extent}x3x{element}>) -> tensor<2x3x{element}> {{
        %s = arith.constant {seed} : {element}
        %seed = tensor.splat %s : tensor<2x3x{element}>
        %r = linalg.reduce ins(%x : tensor<2x{extent}x3x{element}>)
          outs(%seed : tensor<2x3x{element}>) dimensions = [1]
          (%xv: {element}, %acc: {element}) {{
            %v = arith.{arithmetic} %xv, %acc {flags} : {element}
            linalg.yield %v : {element}
          }}
        func.return %r : tensor<2x3x{element}>
      }}
    }}""")


@pytest.mark.parametrize("element,bits", [("i1", 1), ("i8", 8), ("i32", 32), ("i64", 64)])
@pytest.mark.parametrize("arithmetic", ["addi", "muli", "ori", "andi", "maxsi", "minsi", "maxui", "minui"])
def test_singleton_integer_monoid_becomes_exact_indexed_movement(element, bits, arithmetic):
    mask = (1 << bits) - 1
    seed = dict(
        addi=0, muli=1, ori=0, andi=mask, maxsi=1 << (bits - 1), minsi=(1 << (bits - 1)) - 1, maxui=0, minui=mask
    )[arithmetic]
    module = source(element, arithmetic, seed)
    assert fold_singleton_reductions(module) == 1
    module.verify()
    reshaped = next(op for op in module.walk() if op.name == "tensor.reshape")
    assert reshaped.source is next(op for op in module.walk() if op.name == "func.func").body.block.args[0]
    assert reshaped.result.type.get_shape() == (2, 3)
    assert not any(op.name == "linalg.reduce" or op.name == "arith." + arithmetic for op in module.walk())


@pytest.mark.parametrize("extent,seed", [(2, -128), (0, -128), (1, 0)])
def test_non_singleton_or_non_identity_seed_is_retained(extent, seed):
    assert_unchanged(source("i8", "maxsi", seed, extent=extent), fold_singleton_reductions)


@pytest.mark.parametrize("arithmetic,seed", [("maximumf", "-0x7F800000"), ("addf", "0.0")])
def test_float_singleton_does_not_erase_nan_or_signed_zero_semantics(arithmetic, seed):
    # Float extrema can change NaN payloads; adding +0 can change a -0 word.
    assert_unchanged(source("f32", arithmetic, seed), fold_singleton_reductions)


def test_overflow_promises_do_not_gain_a_modular_identity_proof():
    module = source("i8", "addi", 0, flags="overflow<nsw>")
    assert fold_singleton_reductions(module) == 0


@pytest.mark.parametrize("element,bits", [("i1", 1), ("i8", 8), ("i32", 32), ("i64", 64)])
def test_boundary_normalization_preserves_the_closed_integer_identity(element, bits):
    from merlin.llvmlower.structural_folds import normalize_singleton_boundaries

    module = source(element, "maxui", 0)
    assert normalize_singleton_boundaries(module) == 1
    assert not any(op.name in {"linalg.reduce", "linalg.generic", "arith.maxui"} for op in module.walk())


def test_boundary_normalization_retains_a_nonidentity_integer_computation():
    from merlin.llvmlower.structural_folds import normalize_singleton_boundaries

    module = source("i8", "maxsi", 0)
    assert normalize_singleton_boundaries(module) == 1
    assert sum(op.name == "arith.maxsi" for op in module.walk()) == 1
