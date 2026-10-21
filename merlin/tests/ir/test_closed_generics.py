"""Closed elementwise generics become exact uniform fills; nothing else is folded."""

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen.closed_scalar import fold_closed_generics


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
