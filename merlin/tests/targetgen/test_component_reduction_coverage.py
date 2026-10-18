"""A native reduction must offload its numerator, not only its input conversion."""

import json

import numpy as np
import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen.core_aten_components import prepare_integer_components, unlowered_data_reductions


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
    module = parse_mlir_text("""builtin.module {
      func.func @forward(%x: tensor<4xi64>) -> tensor<f32> {
        %e=tensor.empty() : tensor<4xf32>
        %cast=linalg.generic {
          indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]
        } ins(%x : tensor<4xi64>) outs(%e : tensor<4xf32>) {
        ^bb0(%a: i64,%unused: f32):
          %v=arith.sitofp %a : i64 to f32
          linalg.yield %v : f32
        } -> tensor<4xf32>
        %zero=arith.constant 0.0 : f32
        %seed=tensor.splat %zero : tensor<f32>
        %sum=linalg.generic {
          indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->()>],iterator_types=["reduction"]
        } ins(%cast : tensor<4xf32>) outs(%seed : tensor<f32>) {
        ^bb1(%a: f32,%b: f32):
          %nonlinear=math.exp %a : f32
          %v=arith.addf %nonlinear,%b : f32
          linalg.yield %v : f32
        } -> tensor<f32>
        func.return %sum : tensor<f32>
      }}""")
    original = str(module) + "\n"
    (tmp_path / "model.mlir").write_text(original)
    np.savez(tmp_path / "inputs.npz", in0=np.array([1, 2, 3, 4], np.int64))

    class Provider:
        def admit_numeric_schedule(self, *args, **kwargs):
            raise AssertionError("input conversion cannot request a reduction placement proof")

    decision = prepare_integer_components(
        tmp_path,
        dict(cases=[dict(input_abi=[dict(dtype="i64")])]),
        dict(spec={}, host={}),
        Provider(),
        None,
        module,
        limb_attribute="test.limb",
    )
    assert decision is None
    assert (tmp_path / "model.mlir").read_text() == original
    assert (
        json.loads((tmp_path / "exact-dtype-refusal.json").read_text())["reason"]
        == "integer conversion alone is not a device reduction"
    )


def test_device_pointwise_prelude_does_not_hide_a_host_bilinear_loop(tmp_path):
    module = parse_mlir_text("""builtin.module {
      func.func @forward(%a: tensor<2xf16>,%b: tensor<2xf16>) -> tensor<f32> {
        %e=tensor.empty() : tensor<2xf16>
        %pre=linalg.generic {
          indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],
          iterator_types=["parallel"]
        } ins(%a,%b : tensor<2xf16>,tensor<2xf16>) outs(%e : tensor<2xf16>) {
        ^bb0(%x: f16,%y: f16,%unused: f16):
          %v=arith.addf %x,%y : f16
          linalg.yield %v : f16
        } -> tensor<2xf16>
        %out=tensor.empty() : tensor<f32>
        %r=linalg.generic {indexing_maps=[affine_map<()->()>],iterator_types=[]}
          outs(%out : tensor<f32>) {
        ^bb1(%unused: f32):
          %lo=arith.constant 0 : index
          %hi=arith.constant 2 : index
          %step=arith.constant 1 : index
          %zero=arith.constant 0.0 : f32
          %sum=scf.for %k=%lo to %hi step %step iter_args(%s=%zero) -> (f32) {
            %x=tensor.extract %pre[%k] : tensor<2xf16>
            %y=tensor.extract %b[%k] : tensor<2xf16>
            %xf=arith.extf %x : f16 to f32
            %yf=arith.extf %y : f16 to f32
            %p=arith.mulf %xf,%yf : f32
            %cube=arith.mulf %p,%yf : f32
            %v=arith.addf %s,%cube : f32
            scf.yield %v : f32
          }
          linalg.yield %sum : f32
        } -> tensor<f32>
        func.return %r : tensor<f32>
      }}""")
    original = str(module) + "\n"
    (tmp_path / "model.mlir").write_text(original)
    np.savez(tmp_path / "inputs.npz", in0=np.array([1, 2], np.float16), in1=np.array([3, 4], np.float16))

    class Provider:
        def admit_numeric_schedule(self, *args, **kwargs):
            raise AssertionError("a partial prelude must not request placement")

    decision = prepare_integer_components(
        tmp_path,
        dict(cases=[dict(input_abi=[dict(dtype="f16")] * 2)]),
        dict(spec={}, host={}),
        Provider(),
        None,
        module,
        limb_attribute="test.limb",
    )
    assert decision is None
    assert (tmp_path / "model.mlir").read_text() == original
    refusal = json.loads((tmp_path / "exact-dtype-refusal.json").read_text())
    assert "partial preludes" in refusal["reason"] and refusal["source_operation_ordinals"]


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
    module = parse_mlir_text(f"""module {{
      func.func @forward(%x:tensor<2xf32>,%y:tensor<2xf32>,%z:tensor<2xf32>)
        -> (tensor<2xf32>,tensor<2xf32>,tensor<2xf32>,tensor<2xf32>) {{
        %e=tensor.empty() : tensor<2xf32>
        %r=linalg.generic {{indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,
          affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}}
          ins(%x,%y,%z:tensor<2xf32>,tensor<2xf32>,tensor<2xf32>) outs(%e:tensor<2xf32>) {{
          ^bb0(%a:f32,%b:f32,%c:f32,%unused:f32):
            %fma=math.fma %a,%b,%c : f32
            %nan=arith.cmpf {predicate},%a,{right} : f32
            %v=arith.select %nan,{choices} : f32
            linalg.yield %v : f32
        }} -> tensor<2xf32>
        return %r,%x,%y,%z : tensor<2xf32>,tensor<2xf32>,tensor<2xf32>,tensor<2xf32>
      }} }}""")
    (tmp_path / "model.mlir").write_text(str(module) + "\n")
    values = [np.array(v, np.float32) for v in ([1, 2], [3, 4], [5, 6])]
    np.savez(tmp_path / "inputs.npz", **{f"in{i}": value for i, value in enumerate(values)})
    expected = values[0] * values[1] + values[2] if qualified else values[0]
    record = dict(
        status="bundled",
        input_abi=[dict(dtype="f32")] * 3,
        output_indices=[0],
        semantic_boundary=dict(post_indices=[1, 2, 3]),
        native_eager_boundary=dict(
            authority="untransformed_native_cpu_eager",
            output_bytes=[expected.tobytes().hex()],
            post_input_bytes=[v.tobytes().hex() for v in values],
        ),
    )
    called = []

    class Provider:
        def admit_numeric_schedule(self, source, **kwargs):
            called.append(source)
            return dict(status="declined", reason="test grants no device implementation")

    decision = prepare_integer_components(
        tmp_path,
        dict(cases=[record]),
        dict(spec={}, host={}),
        Provider(),
        None,
        module,
        limb_attribute="test.limb",
        native_f32=True,
    )
    assert decision is None  # The provider deliberately withholds its proof.
    assert bool(called) is qualified
    if qualified:
        assert "test.limb" in called[0]
    else:
        assert "partial preludes" in (tmp_path / "exact-dtype-refusal.json").read_text()


def test_nan_classification_after_conversion_has_no_arithmetic_witness():
    from merlin.llvmlower.component_nan_readout import qualify_nan_readout

    module = parse_mlir_text("""module {
      func.func @forward(%x:i64) -> i1 {
        %f=arith.sitofp %x : i64 to f32
        %nan=arith.cmpf uno,%f,%f : f32
        return %nan : i1
      }}""")
    assert qualify_nan_readout(module, [{"source_data_arithmetic": False}]) == []
    assert unlowered_data_reductions(module, "test.limb")
