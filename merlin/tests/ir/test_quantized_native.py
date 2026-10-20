# ruff: noqa: E501 -- MLIR fixture operations keep one source line
"""General quantized fallback: one entry rounding and exact rational readout."""

import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from merlin.common.yaml import load_yaml
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.quantized_native import parameters, quantize_entries

ROOT = Path(__file__).resolve().parents[3]


def contract():
    return parameters(load_yaml(ROOT / "examples/gemmini/target/software-spec.yaml"))


def test_declared_calibration_independent_of_payload():
    p = contract()
    assert Fraction(p["scale_numerator"], p["scale_denominator"]) == Fraction(1, 64)
    assert p["calibration_domain"]["evaluation_inputs"] == "never_read"
    assert p["nonfinite"] == "byte_storage_zero_nan_saturating_infinity"
    s = load_yaml(ROOT / "examples/gemmini/target/software-spec.yaml")
    s["numerical_semantics"]["calibration_domain"]["evaluation_inputs"] = "read"
    with pytest.raises(ValueError):
        parameters(s)


def test_quantized_grading_checks_oracle_bytes_and_retains_eager_diagnostic():
    from merlin.targetgen.core_aten_batch_grade import grade_core_aten_batch

    reference = np.array([Fraction(1, 64)], np.float32).tobytes()
    record = dict(
        overload="fixture",
        status="bundled",
        numeric_policy="spec_int8",
        output_indices=[0],
        output_abi=[dict(dtype="f32", shape=[1])],
        case=dict(expected=dict(kind="tensor", dtype="float32", shape=[1], values=[0.016])),
        numeric_admission=dict(
            quantized_contract=contract(),
            integer_reference=dict(
                status="prepared", engine="independent_source", expected_output_bytes={"0": reference.hex()}
            ),
        ),
    )
    positive = grade_core_aten_batch(dict(cases=[record]), [reference])["cases"]["fixture"]
    assert positive["status"] == "pass" and positive["precision_class"] == "b"
    negative = grade_core_aten_batch(dict(cases=[record]), [np.array([0.016], np.float32).tobytes()])["cases"][
        "fixture"
    ]
    assert negative["status"] == "mismatch"
    assert negative["integer_reference"]["status"] == "mismatch"


def test_full_entry_domain_and_posts_unchanged(tmp_path):
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.pipeline import lower_to_llvm_ir
    from merlin.llvmlower.toolchain import clang

    n = 520
    module = parse_mlir_text(f"""module {{ func.func @forward(%x:tensor<{n}xf32>) -> (tensor<{n}xf32>,tensor<{n}xf32>) attributes {{llvm.emit_c_interface}} {{
      %e=tensor.empty() : tensor<{n}xf32>
      %r=linalg.generic {{indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}}
        ins(%x:tensor<{n}xf32>) outs(%e:tensor<{n}xf32>) {{
      ^bb0(%a:f32,%b:f32): linalg.yield %a : f32
      }} -> tensor<{n}xf32>
      return %r,%x : tensor<{n}xf32>,tensor<{n}xf32>
    }} }}""")
    quantize_entries(module, contract())
    module.verify()
    function = next(op for op in module.walk() if op.name == "func.func")
    ret = function.body.block.last_op
    assert ret.operands[1] is function.body.block.args[0]
    from merlin.xdsl_dialects._common import text

    llvm = lower_to_llvm_ir(text(module, generic=True), workdir=tmp_path / "lower", vectorize=False)
    ll = tmp_path / "model.ll"
    ll.write_text(llvm)
    obj = ll.with_suffix(".o")
    lib = tmp_path / "model.so"
    subprocess.run([str(clang()), "-fPIC", "-O2", "-c", str(ll), "-o", str(obj)], check=True, capture_output=True)
    subprocess.run(
        ["cc", "-fPIC", "-shared", str(obj), str(mlir_runtime_c()), "-lm", "-o", str(lib)],
        check=True,
        capture_output=True,
    )
    x = (np.arange(n, dtype=np.float32) - 260) / 128
    x[:8] = [np.nan, np.inf, -np.inf, -0.0, np.finfo(np.float32).max, np.finfo(np.float32).tiny, 0.5 / 64, 1.5 / 64]
    y = np.zeros(n, np.float32)
    post = np.zeros(n, np.float32)
    HostModel.load(str(lib))([(a.ctypes.data, a.shape) for a in (x, y, post)])
    from merlin.targetgen.exact_arithmetic import quantize

    expected = np.asarray(
        (quantize(x, Fraction(1, 64), nonfinite=contract()["nonfinite"]) * Fraction(1, 64)).tolist(), np.float32
    )
    assert y.tobytes() == expected.tobytes()
    assert post.tobytes() == x.tobytes()


def test_gaussian_reference_is_rational():
    from merlin.targetgen.quantized_native_reference import Gaussian

    a, b = Gaussian(Fraction(1, 3), Fraction(7, 11)), Gaussian(2, -3)
    p = a * b
    assert p.real == Fraction(85, 33) and p.imag == Fraction(3, 11)
    q = p / b
    assert q.real == a.real and q.imag == a.imag


def test_rational_transform_oracle_has_no_trigonometric_approximation():
    from merlin.targetgen.exact_arithmetic import ExactTensor
    from merlin.targetgen.quantized_native_reference import rational_real_transform

    x = ExactTensor(np.array([[Fraction(i, 3) for i in range(4)]], object), "f32")
    result = rational_real_transform([x, [1], 0, True], lambda v: v.values, ExactTensor).values
    assert [(v.real, v.imag) for v in result.flat] == [(2, 0), (Fraction(-2, 3), Fraction(2, 3)), (Fraction(-2, 3), 0)]
    bad = ExactTensor(np.array([Fraction(1)] * 3, object), "f32")
    with pytest.raises(ValueError, match="non-rational root"):
        rational_real_transform([bad, [0], 0, True], lambda v: v.values, ExactTensor)


def test_runtime_parallel_products_do_not_gain_native_admission(tmp_path, monkeypatch):
    import merlin.targetgen.core_aten_spec_int8 as spec
    from merlin.targetgen.quantized_native import prepare

    software = load_yaml(ROOT / "examples/gemmini/target/software-spec.yaml")
    monkeypatch.setattr(spec, "select_spec", lambda *a: dict(spec=software))
    (tmp_path / "model.mlir").write_text("""module {
      func.func @forward(%x:tensor<2xf32>,%y:tensor<2xf32>) -> tensor<2xf32> {
        %e=tensor.empty() : tensor<2xf32>
        %r=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
          ins(%x,%y:tensor<2xf32>,tensor<2xf32>) outs(%e:tensor<2xf32>) {
        ^bb0(%a:f32,%b:f32,%out:f32):
          %v=arith.mulf %a,%b : f32
          linalg.yield %v : f32
        } -> tensor<2xf32>
        return %r : tensor<2xf32>
    }}""")
    original = (tmp_path / "model.mlir").read_bytes()
    report = dict(cases=[dict(status="bundled", input_abi=[dict(dtype="f32")] * 2, output_abi=[dict(dtype="f32")])])
    assert prepare(tmp_path, report, target="test", facts={}, software_spec=None, host_capabilities=None) is None
    assert (tmp_path / "model.mlir").read_bytes() == original


def test_reduction_dependent_parallel_product_is_still_a_composition(tmp_path, monkeypatch):
    import merlin.targetgen.core_aten_spec_int8 as spec
    from merlin.targetgen.quantized_native import prepare

    software = load_yaml(ROOT / "examples/gemmini/target/software-spec.yaml")
    monkeypatch.setattr(spec, "select_spec", lambda *a: dict(spec=software))
    (tmp_path / "model.mlir").write_text("""module {
      func.func @forward(%x:tensor<2xf32>) -> tensor<2xf32> {
        %z=arith.constant 0.0 : f32
        %seed=tensor.splat %z : tensor<1xf32>
        %sum=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(0)>],iterator_types=["reduction"]}
          ins(%x:tensor<2xf32>) outs(%seed:tensor<1xf32>) {
        ^bb0(%a:f32,%acc:f32):
          %s=arith.addf %a,%acc : f32
          linalg.yield %s : f32
        } -> tensor<1xf32>
        %e=tensor.empty() : tensor<2xf32>
        %r=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
          ins(%x,%sum:tensor<2xf32>,tensor<1xf32>) outs(%e:tensor<2xf32>) {
        ^bb0(%a:f32,%b:f32,%out:f32):
          %v=arith.mulf %a,%b : f32
          linalg.yield %v : f32
        } -> tensor<2xf32>
        return %r : tensor<2xf32>
    }}""")
    report = dict(cases=[dict(status="bundled", input_abi=[dict(dtype="f32")], output_abi=[dict(dtype="f32")])])
    assert prepare(tmp_path, report, target="test", facts={}, software_spec=None, host_capabilities=None) is None
    assert "parallel runtime product" in (tmp_path / "quantized-native-refusal.json").read_text()


def test_native_exact_success_does_not_try_quantization(tmp_path, monkeypatch):
    import merlin.targetgen.core_aten_admission as admission
    import merlin.targetgen.quantized_native as fallback
    from merlin.targetgen.core_aten_numeric import prepare_spec_int8

    expected = dict(native_admission=dict(status="device"))
    monkeypatch.setattr(admission, "prepare", lambda *args, **kw: expected)

    def forbidden(*args, **kw):
        raise AssertionError("native exact path changed")

    monkeypatch.setattr(fallback, "prepare", forbidden)
    assert prepare_spec_int8(tmp_path, {}) is expected


@pytest.mark.parametrize("oracle_failure", [False, True])
def test_payload_changes_cannot_choose_host_placement(tmp_path, monkeypatch, oracle_failure):
    import json
    from types import SimpleNamespace

    import merlin.targetgen.core_aten_components as components
    import merlin.targetgen.core_aten_numeric as numeric
    import merlin.targetgen.core_aten_spec_int8 as spec
    from merlin.targetgen.quantized_native import prepare

    software = load_yaml(ROOT / "examples/gemmini/target/software-spec.yaml")
    monkeypatch.setattr(spec, "select_spec", lambda *a: dict(spec=software))
    prepared_sources = []

    def component_plan(directory, report, selection, provider, package, module, **kw):
        prepared_sources.append(str(module))
        return dict(status="admitted", exact_dtype_sites=1, quantized_contract=kw["quantized_contract"])

    def independent_reference(directory, mode):
        if oracle_failure:
            return SimpleNamespace(returncode=1, stderr="oracle qualification failed")
        request = json.loads((directory / "quantized-native-request.json").read_text())
        value = request["records"][0]["case"]["values"][0]
        (directory / "quantized-native-reference.json").write_text(
            json.dumps(dict(expected_output_bytes={"0": np.array([value], np.float32).tobytes().hex()}))
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(components, "prepare_integer_components", component_plan)
    monkeypatch.setattr(numeric, "_source_reference", independent_reference)
    plans = []
    for ordinal, values in enumerate(([0.0], [float("inf")], [float("nan")])):
        directory = tmp_path / str(ordinal)
        directory.mkdir()
        source = "module { func.func @forward(%x:tensor<1xf32>) -> tensor<1xf32> { return %x : tensor<1xf32> } }"
        (directory / "model.mlir").write_text(source)
        report = dict(
            output_count=1,
            cases=[
                dict(
                    status="bundled",
                    input_abi=[dict(dtype="f32")],
                    output_abi=[dict(dtype="f32")],
                    case=dict(values=values),
                )
            ],
        )
        kwargs = dict(target="test", facts={}, software_spec=None, host_capabilities=None)
        if oracle_failure:
            with pytest.raises(ArithmeticError, match="qualification failed"):
                prepare(directory, report, **kwargs)
            assert (directory / "model.mlir").read_text() == source
        else:
            plan = prepare(directory, report, **kwargs)
            assert plan["native_admission"]["status"] == "device"
            plans.append({k: v for k, v in plan.items() if k != "integer_reference"})
    assert len(set(prepared_sources)) == 1
    if plans:
        assert plans[0] == plans[1] == plans[2]


def test_nan_word_overlay_is_removed_but_live_nan_is_refused():
    from merlin.llvmlower.quantized_native import finite_source

    def fixture(yielded):
        return parse_mlir_text(
            """module {
          func.func @forward(%x:tensor<1xf32>) -> tensor<1xf32> {
            %e=tensor.empty() : tensor<1xf32>
            %r=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
              ins(%x:tensor<1xf32>) outs(%e:tensor<1xf32>) {
            ^bb0(%a:f32,%b:f32):
              %nan=arith.constant 0x7FC00000 : f32
              %isnan=arith.cmpf uno,%a,%a : f32
              %v=arith.select %isnan,%nan,%a : f32
              linalg.yield """
            + yielded
            + """ : f32
            } -> tensor<1xf32>
            return %r : tensor<1xf32>
          }}"""
        )

    module = fixture("%v")
    finite_source(module)
    module.verify()
    assert not any(op.name == "arith.cmpf" for op in module.walk())
    with pytest.raises(ValueError, match="nonfinite literal"):
        finite_source(fixture("%nan"))


def test_rational_divisors_survive_layouts_until_exported_quantize(tmp_path):
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.floating_components import rewrite_floating_components
    from merlin.llvmlower.pipeline import lower_to_llvm_ir
    from merlin.llvmlower.toolchain import clang

    module = parse_mlir_text("""module {
      func.func @forward(%x:tensor<256xf32>) -> tensor<16x16xf32> attributes {llvm.emit_c_interface} {
        %three=arith.constant 3.0 : f32
        %divisor=tensor.splat %three : tensor<256xf32>
        %e=tensor.empty() : tensor<256xf32>
        %q=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
          ins(%x,%divisor:tensor<256xf32>,tensor<256xf32>) outs(%e:tensor<256xf32>) {
        ^bb0(%a:f32,%b:f32,%out:f32):
          %v=arith.divf %a,%b : f32
          linalg.yield %v : f32
        } -> tensor<256xf32>
        %shape=arith.constant dense<[16,16]> : tensor<2xi64>
        %view=tensor.reshape %q(%shape) : (tensor<256xf32>,tensor<2xi64>) -> tensor<16x16xf32>
        %dst=tensor.empty() : tensor<16x16xf32>
        %r=linalg.generic {indexing_maps=[affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>],iterator_types=["parallel","parallel"]}
          ins(%view,%view:tensor<16x16xf32>,tensor<16x16xf32>) outs(%dst:tensor<16x16xf32>) {
        ^bb1(%a:f32,%b:f32,%out:f32):
          %v=arith.addf %a,%b : f32
          linalg.yield %v : f32
        } -> tensor<16x16xf32>
        return %r : tensor<16x16xf32>
      }}""")
    p = contract()
    grids = quantize_entries(module, p)
    bounds, proofs = rewrite_floating_components(
        module, [None], known_grids=grids, allow_reassociation=True, quantized_contract=p
    )
    module.verify()
    assert any(row["denominator"] == 3 for row in proofs)
    from merlin.xdsl_dialects._common import text

    llvm = lower_to_llvm_ir(text(module, generic=True), workdir=tmp_path / "lower", vectorize=False)
    ll = tmp_path / "rational.ll"
    ll.write_text(llvm)
    obj = ll.with_suffix(".o")
    lib = tmp_path / "rational.so"
    subprocess.run([str(clang()), "-fPIC", "-O2", "-c", str(ll), "-o", str(obj)], check=True, capture_output=True)
    subprocess.run(
        ["cc", "-fPIC", "-shared", str(obj), str(mlir_runtime_c()), "-lm", "-o", str(lib)],
        check=True,
        capture_output=True,
    )
    x = np.arange(-128, 128, dtype=np.float32) / 64
    y = np.zeros((16, 16), np.float32)
    HostModel.load(str(lib))([(a.ctypes.data, a.shape) for a in (x, y)])
    expected = np.asarray(
        [min(127, max(-128, round(Fraction(2 * i, 3)))) / 64 for i in range(-128, 128)], np.float32
    ).reshape(16, 16)
    assert y.tobytes() == expected.tobytes()
