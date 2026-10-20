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
    llvm = lower_to_llvm_ir(str(module), workdir=tmp_path / "lower", vectorize=False)
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
