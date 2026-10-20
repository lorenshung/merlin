"""Every integer data producer needs its own native-unit binding."""

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen.native_integer_work import uncovered


def source():
    return parse_mlir_text("""module {func.func @forward(%x:tensor<3xi32>,%y:tensor<3xi32>) -> tensor<i32> {
      %e=tensor.empty() : tensor<3xi32>
      %sum=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
        ins(%x,%y:tensor<3xi32>,tensor<3xi32>) outs(%e:tensor<3xi32>) {
        ^bb0(%a:i32,%b:i32,%unused:i32):
        %v=arith.addi %a,%b : i32
        linalg.yield %v : i32
      } -> tensor<3xi32>
      %zero=arith.constant 0 : i32
      %seed=tensor.splat %zero : tensor<i32>
      %reduced=linalg.reduce ins(%sum:tensor<3xi32>) outs(%seed:tensor<i32>) dimensions=[0]
        (%v:i32,%acc:i32) { %added=arith.addi %v,%acc : i32
        linalg.yield %added : i32 }
      return %reduced : tensor<i32>
    }}""")


def test_both_parallel_fold_and_bias_reduction_need_a_binding():
    module = source()
    ordinals = [i for i, op in enumerate(module.walk()) if op.name in {"linalg.generic", "linalg.reduce"}]
    assert uncovered(module, None) == ordinals
    assert uncovered(module, dict(bindings=[dict(source_operation_ordinal=ordinals[0])])) == ordinals[1:]
    assert uncovered(module, dict(bindings=[dict(source_operation_ordinal=i) for i in ordinals])) == []


def test_float_readout_integer_representation_is_separate_from_source_integer_work():
    module = parse_mlir_text("""module {func.func @forward(%x:tensor<3xi32>) -> tensor<3xf32> {
      %e=tensor.empty() : tensor<3xf32>
      %out=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
        ins(%x:tensor<3xi32>) outs(%e:tensor<3xf32>) {
        ^bb0(%a:i32,%unused:f32):
        %zero=arith.constant 0 : i32
        %v=arith.addi %a,%zero : i32
        %f=arith.sitofp %v : i32 to f32
        linalg.yield %f : f32
      } -> tensor<3xf32>
      return %out : tensor<3xf32>
    }}""")
    assert uncovered(module, None) == []
