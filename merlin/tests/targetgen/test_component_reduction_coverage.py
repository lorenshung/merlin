"""A native reduction must offload its numerator, not only its input conversion."""

import numpy as np
import pytest
from mlir_fixtures import generic, pointwise
from targetgen_fixtures import DecliningProvider, assert_host_refusal, eager_record

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen.core_aten_components import unlowered_data_reductions


@pytest.mark.parametrize("intrinsic", ["exp", "sin", "cos", "log", "rsqrt"])
def test_host_math_after_conversion_is_not_device_operation_coverage(intrinsic):
    module = parse_mlir_text(f"""builtin.module {{
      func.func @forward(%a: i64) -> f32 {{
        %c = arith.sitofp %a : i64 to f32
        %v = math.{intrinsic} %c : f32
        func.return %v : f32
      }}
    }}""")
    missed = unlowered_data_reductions(module, "test.limb")
    assert [list(module.walk())[i].name for i in missed] == [f"math.{intrinsic}"]


def test_integer_conversion_does_not_admit_a_retained_host_float_sum(tmp_path):
    cast = generic([("%x", "tensor<4xi64>")], "tensor<4xf32>", "%v=arith.sitofp %a : i64 to f32", name="%cast")
    module = parse_mlir_text(f"""builtin.module {{
      func.func @forward(%x: tensor<4xi64>) -> tensor<f32> {{
        {cast}
        %zero=arith.constant 0.0 : f32
        %seed=tensor.splat %zero : tensor<f32>
        %sum=linalg.generic {{
          indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->()>],iterator_types=["reduction"]
        }} ins(%cast : tensor<4xf32>) outs(%seed : tensor<f32>) {{
        ^bb1(%a: f32,%b: f32):
          %nonlinear=math.exp %a : f32
          %v=arith.addf %nonlinear,%b : f32
          linalg.yield %v : f32
        }} -> tensor<f32>
        func.return %sum : tensor<f32>
      }}}}""")
    inputs, cases = [np.array([1, 2, 3, 4], np.int64)], [dict(input_abi=[dict(dtype="i64")])]
    assert_host_refusal(tmp_path, module, inputs, cases, "native_aggregate_nan_order_unproved")


def test_device_pointwise_prelude_does_not_hide_a_host_bilinear_loop(tmp_path):
    ins = [("%a", "tensor<2xf16>"), ("%b", "tensor<2xf16>")]
    prelude = generic(ins, "tensor<2xf16>", "%v=arith.addf %x,%y : f16", name="%pre", args="xy")
    module = parse_mlir_text(f"""builtin.module {{
      func.func @forward(%a: tensor<2xf16>,%b: tensor<2xf16>) -> tensor<f32> {{
        {prelude}
        %out=tensor.empty() : tensor<f32>
        %r=linalg.generic {{indexing_maps=[affine_map<()->()>],iterator_types=[]}}
          outs(%out : tensor<f32>) {{
        ^bb1(%unused: f32):
          %lo=arith.constant 0 : index
          %hi=arith.constant 2 : index
          %step=arith.constant 1 : index
          %zero=arith.constant 0.0 : f32
          %sum=scf.for %k=%lo to %hi step %step iter_args(%s=%zero) -> (f32) {{
            %x=tensor.extract %pre[%k] : tensor<2xf16>
            %y=tensor.extract %b[%k] : tensor<2xf16>
            %xf=arith.extf %x : f16 to f32
            %yf=arith.extf %y : f16 to f32
            %p=arith.mulf %xf,%yf : f32
            %cube=arith.mulf %p,%yf : f32
            %v=arith.addf %s,%cube : f32
            scf.yield %v : f32
          }}
          linalg.yield %sum : f32
        }} -> tensor<f32>
        func.return %r : tensor<f32>
      }}}}""")
    inputs = [np.array([1, 2], np.float16), np.array([3, 4], np.float16)]
    cases = [dict(input_abi=[dict(dtype="f16")] * 2)]
    assert_host_refusal(tmp_path, module, inputs, cases, "native_competing_nan_priority_unproved")


@pytest.mark.parametrize("predicate", ["cmpf", "maximumf", "minimumf"])
def test_a_host_predicate_after_an_integer_conversion_is_not_device_arithmetic(predicate):
    result_type = "i1" if predicate == "cmpf" else "f32"
    expression = "arith.cmpf oeq, %c, %zero : f32" if predicate == "cmpf" else f"arith.{predicate} %c, %zero : f32"
    module = parse_mlir_text(f"""module {{
      func.func @forward(%x: i64) -> {result_type} {{
        %c=arith.sitofp %x : i64 to f32
        %zero=arith.constant 0.0 : f32
        %v={expression}
        func.return %v : {result_type}
      }} }}""")
    missed = unlowered_data_reductions(module, "test.limb")
    assert [list(module.walk())[i].name for i in missed] == ["arith." + predicate]


def test_an_unqualified_integer_comparison_remains_host():
    module = parse_mlir_text("""module {
      func.func @forward(%x: i32) -> i1 {
        %zero=arith.constant 0 : i32
        %v=arith.cmpi eq, %x, %zero : i32
        func.return %v : i1
      }}""")
    assert unlowered_data_reductions(module, "test.limb")


def test_address_predicates_do_not_claim_remaining_numerical_arithmetic():
    module = parse_mlir_text("""module {
      func.func @forward(%i: index,%n: index) -> i1 {
        %r=arith.cmpi ult, %i,%n : index
        return %r : i1
      }}""")
    assert unlowered_data_reductions(module, "test.limb") == []


@pytest.mark.parametrize("predicate,qualified", [("uno", True), ("ord", True), ("olt", False)])
def test_fused_components_retain_only_self_nan_classification(tmp_path, monkeypatch, predicate, qualified):
    monkeypatch.setattr(
        "merlin.targetgen.transfer_contracts.screen_transfer_contract", lambda *args, **kwargs: dict(status="admitted")
    )
    right = "%a" if qualified else "%b"
    choices = "%fma, %a" if predicate == "ord" else "%a, %fma"
    body = f"""%fma=math.fma %a,%b,%c : f32
      %nan=arith.cmpf {predicate},%a,{right} : f32
      %v=arith.select %nan,{choices} : f32"""
    computation = generic([(v, "tensor<2xf32>") for v in ("%x", "%y", "%z")], "tensor<2xf32>", body)
    module = parse_mlir_text(f"""module {{
      func.func @forward(%x:tensor<2xf32>,%y:tensor<2xf32>,%z:tensor<2xf32>)
        -> (tensor<2xf32>,tensor<2xf32>,tensor<2xf32>,tensor<2xf32>) {{
        {computation}
        return %r,%x,%y,%z : tensor<2xf32>,tensor<2xf32>,tensor<2xf32>,tensor<2xf32>
      }} }}""")
    values = [np.array(v, np.float32) for v in ([1, 2], [3, 4], [5, 6])]
    expected = values[0] * values[1] + values[2] if qualified else values[0]
    hexes = [v.tobytes().hex() for v in values]
    record = eager_record([dict(dtype="f32")] * 3, [expected.tobytes().hex()], hexes, [1, 2, 3])
    provider = DecliningProvider()  # It would withhold its proof; it must not even be asked.
    code = "native_parallel_runtime_product_construction"
    assert_host_refusal(tmp_path, module, values, [record], code, provider)
    assert not provider.called


def test_nan_classification_after_conversion_has_no_arithmetic_witness():
    module = parse_mlir_text("""module {
      func.func @forward(%x:i64) -> i1 {
        %f=arith.sitofp %x : i64 to f32
        %nan=arith.cmpf uno,%f,%f : f32
        return %nan : i1
      }}""")
    assert unlowered_data_reductions(module, "test.limb")


@pytest.mark.parametrize("width", [8, 32, 64])
def test_storage_copy_does_not_cover_a_retained_host_threshold(tmp_path, width):
    typ = f"tensor<4xi{width}>"
    body = f"%z=arith.constant 0 : i{width}\n%p=arith.cmpi slt,%a,%z : i{width}\n%v=arith.select %p,%z,%a : i{width}"
    inputs, cases = [np.array([-1, 0, 1, 127], dtype=f"int{width}")], [dict(input_abi=[dict(dtype=f"i{width}")])]
    code = "native_family_arithmetic_retained_host"
    assert_host_refusal(tmp_path, pointwise([typ], typ, body), inputs, cases, code)
