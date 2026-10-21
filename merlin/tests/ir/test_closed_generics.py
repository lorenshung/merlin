"""Closed elementwise generics become exact uniform fills; nothing else is folded."""

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen.closed_scalar import fold_closed_generics, fold_unit_scales


def program(body, ins="%c", types="tensor<bf16>", source="tensor<f32>"):
    return parse_mlir_text(f"""module {{ func.func @forward(%x: tensor<f32>) -> tensor<bf16> {{
      %z = arith.constant 0.0 : f32
      %c = tensor.splat %z : tensor<f32>
      %e = tensor.empty() : tensor<bf16>
      %r = linalg.generic {{indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []}}
        ins({ins} : {source}) outs(%e : {types}) {{
        ^bb0(%a: f32, %u: bf16):
          {body}
      }} -> {types}
      func.return %r : {types}
    }} }}""")


def test_empty_reduction_narrowed_to_storage_is_a_literal_fill():
    module = program("%n = arith.truncf %a : f32 to bf16\nlinalg.yield %n : bf16")
    assert fold_closed_generics(module) == 1
    module.verify()
    returned = next(op for op in module.walk() if op.name == "func.return")
    assert returned.operands[0].owner.name == "tensor.splat"
    assert not any(op.name == "linalg.generic" for op in module.walk())


def test_runtime_input_is_not_folded():
    module = program("%n = arith.truncf %a : f32 to bf16\nlinalg.yield %n : bf16", ins="%x")
    assert fold_closed_generics(module) == 0


def test_non_finite_closed_result_is_not_folded():
    module = program("%d = arith.divf %a, %a : f32\n%n = arith.truncf %d : f32 to bf16\nlinalg.yield %n : bf16")
    assert fold_closed_generics(module) == 0


def test_unit_scale_is_an_exact_identity():
    for body, folded in (
        ("%k = arith.constant 1.0 : f32\n%m = arith.mulf %k, %a : f32\n%n = arith.truncf %m : f32 to bf16\nlinalg.yield %n : bf16", 1),
        ("%k = arith.constant 1.0 : f32\n%m = arith.divf %a, %k : f32\n%n = arith.truncf %m : f32 to bf16\nlinalg.yield %n : bf16", 1),
        ("%k = arith.constant 1.0 : f32\n%m = arith.divf %k, %a : f32\n%n = arith.truncf %m : f32 to bf16\nlinalg.yield %n : bf16", 0),
        ("%k = arith.constant 2.0 : f32\n%m = arith.mulf %a, %k : f32\n%n = arith.truncf %m : f32 to bf16\nlinalg.yield %n : bf16", 0),
        ("%k = arith.constant -1.0 : f32\n%m = arith.mulf %a, %k : f32\n%n = arith.truncf %m : f32 to bf16\nlinalg.yield %n : bf16", 0),
    ):
        module = program(body, ins="%x")
        assert fold_unit_scales(module) == folded
        module.verify()


def test_round_trip_through_a_copied_unit_scale_is_the_input():
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.targetgen.core_aten_numeric import normalize_closed_identities

    # Frontend form of f16 x * 1.0: widen (and scale) in one generic, narrow in the next.
    module = parse_mlir_text("""module { func.func @forward(%x: tensor<4xf16>) -> tensor<4xf16> {
      %one = arith.constant 1.0 : f32
      %s = tensor.splat %one : tensor<4xf32>
      %e = tensor.empty() : tensor<4xf32>
      %w = linalg.generic {indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>, affine_map<(i)->(i)>], iterator_types = ["parallel"]}
        ins(%x, %s : tensor<4xf16>, tensor<4xf32>) outs(%e : tensor<4xf32>) {
        ^bb0(%a: f16, %b: f32, %u: f32):
          %c = arith.extf %a : f16 to f32
          %m = arith.mulf %c, %b : f32
          linalg.yield %m : f32
      } -> tensor<4xf32>
      %e2 = tensor.empty() : tensor<4xf16>
      %n = linalg.generic {indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>], iterator_types = ["parallel"]}
        ins(%w : tensor<4xf32>) outs(%e2 : tensor<4xf16>) {
        ^bb0(%a: f32, %u: f16):
          %t = arith.truncf %a : f32 to f16
          linalg.yield %t : f16
      } -> tensor<4xf16>
      func.return %n : tensor<4xf16>
    } }""")
    normalize_closed_identities(module)
    returned = next(op for op in module.walk() if op.name == "func.return")
    assert returned.operands[0] is next(op for op in module.walk() if op.name == "func.func").body.block.args[0]
