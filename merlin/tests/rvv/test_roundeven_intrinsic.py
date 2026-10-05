"""The compact exact round-even lowering is a real, correctly placed compiler lever."""

import pytest

from merlin.llvmlower import impr_features as F
from merlin.llvmlower.pipeline import build_rvv_pipeline, lower_to_llvm_ir
from merlin.llvmlower.roundeven_intrinsic import FEATURE, MARKER, apply_for_test, ensure_registered


def test_rewrite_changes_only_roundeven_to_the_exact_llvm_intrinsic():
    src = """module {
      func.func @f(%x: f32) -> (f32, f32) {
        %r = math.roundeven %x : f32
        %e = math.exp %x : f32
        return %r, %e : f32, f32
      }
    }"""
    got, count = apply_for_test(src)
    assert count == 1
    assert "llvm.intr.roundeven" in got
    assert "math.roundeven" not in got
    assert "math.exp" in got, "unrelated math operations must retain the established libm path"


def test_feature_places_the_runner_marker_after_linalg_becomes_loops():
    ensure_registered()
    passes = F.apply_pipeline(["canonicalize", "func.func(convert-linalg-to-loops)", "convert-scf-to-cf"], {FEATURE})
    assert passes.index(MARKER) == passes.index("func.func(convert-linalg-to-loops)") + 1
    assert passes.index(MARKER) < passes.index("convert-scf-to-cf")


def test_real_lowering_reaches_llvm_intrinsic_not_libm(tmp_path):
    src = """module {
      func.func @f(%a: memref<16xf32>, %b: memref<16xf32>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c16 = arith.constant 16 : index
        scf.for %i = %c0 to %c16 step %c1 {
          %x = memref.load %a[%i] : memref<16xf32>
          %r = math.roundeven %x : f32
          memref.store %r, %b[%i] : memref<16xf32>
        }
        return
      }
    }"""
    got = lower_to_llvm_ir(src, workdir=tmp_path, features={FEATURE})
    assert "@llvm.roundeven.f32" in got
    assert "roundevenf" not in got


def test_roundeven_intrinsic_and_arithmetic_expansion_are_explicit_alternatives(tmp_path):
    ensure_registered()
    with pytest.raises(Exception, match="alternative exact lowerings"):
        lower_to_llvm_ir(
            "module {}", workdir=tmp_path, vectorize=True, features={FEATURE, "fuse_quantize_round_convert"}
        )


def test_every_exact_math_op_is_inlined_and_transcendentals_keep_libm(tmp_path):
    from merlin.llvmlower.roundeven_intrinsic import EXACT_FEATURE

    body = "\n".join(
        f"          %{name} = math.{name} %x : f64\n          memref.store %{name}, %b[%i] : memref<16xf64>"
        for name in ("floor", "ceil", "trunc", "round", "roundeven", "absf", "exp")
    )
    src = f"""module {{
      func.func @f(%a: memref<16xf64>, %b: memref<16xf64>) {{
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c16 = arith.constant 16 : index
        scf.for %i = %c0 to %c16 step %c1 {{
          %x = memref.load %a[%i] : memref<16xf64>
{body}
        }}
        return
      }}
    }}"""
    ensure_registered()
    got = lower_to_llvm_ir(src, workdir=tmp_path, features={EXACT_FEATURE})
    for intrinsic in ("floor", "ceil", "trunc", "round", "roundeven", "fabs"):
        assert f"@llvm.{intrinsic}.f64" in got, intrinsic
    for libm in ("@floor(", "@ceil(", "@trunc(", "@round(", "@roundeven(", "@fabs("):
        assert libm not in got, libm
    assert "@exp(" in got, "a transcendental keeps the established libm call"


def test_the_two_intrinsic_levers_compose(tmp_path):
    from merlin.llvmlower.roundeven_intrinsic import EXACT_FEATURE

    src = """module {
      func.func @f(%a: memref<4xf32>, %b: memref<4xf32>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        scf.for %i = %c0 to %c4 step %c1 {
          %x = memref.load %a[%i] : memref<4xf32>
          %r = math.roundeven %x : f32
          %f = math.floor %r : f32
          memref.store %f, %b[%i] : memref<4xf32>
        }
        return
      }
    }"""
    ensure_registered()
    got = lower_to_llvm_ir(src, workdir=tmp_path, features={FEATURE, EXACT_FEATURE})
    assert "@llvm.roundeven.f32" in got and "@llvm.floor.f32" in got


def _pow2_by_bits(i: int, width: int) -> float:
    """The arithmetic the pow-of-two rewrite emits, re-done in numpy for one integer exponent."""
    import numpy as np

    bias, lo, hi, mantissa = {64: (1023, -1022, 1023, 52), 32: (127, -126, 127, 23)}[width]
    as_float = np.float64 if width == 64 else np.float32
    as_int = np.int64 if width == 64 else np.int32

    def power(e: int):
        return np.array([(e + bias) << mantissa], dtype=as_int).view(as_float)[0]

    e1 = max(min(i, hi), lo)
    e2 = max(min(i - e1, hi), lo)
    with np.errstate(over="ignore", under="ignore"):
        return power(e1) * power(e2)


@pytest.mark.parametrize("width", [64, 32])
def test_the_power_of_two_construction_is_the_correctly_rounded_power(width):
    import math

    import numpy as np

    as_float = np.float64 if width == 64 else np.float32
    for i in range(-1200 if width == 64 else -200, 1100 if width == 64 else 200):
        want = as_float(math.ldexp(1.0, i)) if i < (1024 if width == 64 else 128) else as_float(np.inf)
        got = _pow2_by_bits(i, width)
        assert got.tobytes() == as_float(want).tobytes(), (width, i)


def test_pow_of_two_of_an_integer_is_inlined_and_other_pows_keep_libm(tmp_path):
    from merlin.llvmlower.roundeven_intrinsic import EXACT_FEATURE

    src = """module {
      func.func @f(%a: memref<8xi64>, %x: memref<8xf64>, %b: memref<8xf64>, %c: memref<8xf64>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c8 = arith.constant 8 : index
        %two = arith.constant 2.0 : f64
        %three = arith.constant 3.0 : f64
        scf.for %i = %c0 to %c8 step %c1 {
          %k = memref.load %a[%i] : memref<8xi64>
          %e = arith.sitofp %k : i64 to f64
          %p = math.powf %two, %e : f64
          memref.store %p, %b[%i] : memref<8xf64>
          %y = memref.load %x[%i] : memref<8xf64>
          %q = math.powf %three, %y : f64
          memref.store %q, %c[%i] : memref<8xf64>
        }
        return
      }
    }"""
    ensure_registered()
    got = lower_to_llvm_ir(src, workdir=tmp_path, features={EXACT_FEATURE})
    assert got.count("@pow(") == 2, "only the non-power-of-two pow keeps its libm call (declaration + call)"
    assert "bitcast i64" in got and "shl i64" in got


_BF16_REQUANT = """#map = affine_map<(d0) -> (d0)>
module {
  func.func @forward(%a: tensor<64xi32>, %s: tensor<64xbf16>) -> tensor<64xi8> {
    %e = tensor.empty() : tensor<64xi8>
    %r = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
        ins(%a, %s : tensor<64xi32>, tensor<64xbf16>) outs(%e : tensor<64xi8>) {
    ^bb0(%x: i32, %k: bf16, %o: i8):
      %f = arith.sitofp %x : i32 to bf16
      %m = arith.mulf %f, %k : bf16
      %q = math.roundeven %m : bf16
      %lo = arith.constant -1.280000e+02 : bf16
      %hi = arith.constant 1.270000e+02 : bf16
      %c = arith.maximumf %q, %lo : bf16
      %d = arith.minimumf %c, %hi : bf16
      %n = arith.negf %d : bf16
      %y = arith.fptosi %n : bf16 to i8
      linalg.yield %y : i8
    } -> tensor<64xi8>
    return %r : tensor<64xi8>
  }
}"""


def test_bf16_ops_compute_in_f32_and_round_without_a_call(tmp_path):
    """A bf16 requant chain reaches LLVM as f32 arithmetic and integer rounding: no op on bfloat is
    left for LLVM to legalize into a ``__truncsfbf2`` call per element, and no libm call."""
    from merlin.llvmlower.lower import lower_model
    from merlin.llvmlower.roundeven_intrinsic import EXACT_FEATURE

    ensure_registered()
    got = lower_model(
        _BF16_REQUANT, tmp_path / "on", targets=(), textual=True, features=frozenset({EXACT_FEATURE})
    ).ll_path.read_text()
    for kind in (
        "fmul bfloat",
        "fptrunc float",
        "to bfloat\n",
        "fneg bfloat",
        "fpext bfloat",
        "@llvm.roundeven.bf16",
        "@llvm.maximum.bf16",
        "@llvm.minimum.bf16",
        "__truncsfbf2",
        "@roundevenf(",
    ):
        assert kind not in got, kind
    assert "fmul float" in got and "@llvm.roundeven.f32" in got
    base = lower_model(_BF16_REQUANT, tmp_path / "off", targets=(), textual=True).ll_path.read_text()
    assert "fmul bfloat" in base, "without the feature the bf16 arithmetic is LLVM's to legalize"


def _truncsfbf2(bits):
    """``merlin/runtime/abi/mlir_runtime.c``'s f32 -> bf16 rounding, on uint32 bit patterns."""
    import numpy as np

    bits = bits.astype(np.uint64)
    top = bits >> np.uint64(16)
    special = (bits & np.uint64(0x7F800000)) == np.uint64(0x7F800000)
    nan = top | np.where((bits & np.uint64(0x007FFFFF)) != 0, np.uint64(0x40), np.uint64(0))
    rounded = (bits + np.uint64(0x7FFF) + (top & np.uint64(1))) >> np.uint64(16)
    return np.where(special, nan, rounded).astype(np.uint16)


def _host_toolchain():
    from merlin.llvmlower import toolchain

    return toolchain.available()


def _run_unary(tmp_path, tag, body, in_t, out_t, inputs, out_dtype):
    """Lower ``body`` (one element of ``in_t`` -> ``out_t``) with the exact feature for the host, run it."""
    import numpy as np

    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.lower import lower_model
    from merlin.llvmlower.roundeven_intrinsic import EXACT_FEATURE

    n = inputs.size
    src = f"""#map = affine_map<(d0) -> (d0)>
module {{
  func.func @forward(%a: tensor<{n}x{in_t}>) -> tensor<{n}x{out_t}> {{
    %e = tensor.empty() : tensor<{n}x{out_t}>
    %r = linalg.generic {{indexing_maps = [#map, #map], iterator_types = ["parallel"]}}
        ins(%a : tensor<{n}x{in_t}>) outs(%e : tensor<{n}x{out_t}>) {{
    ^bb0(%x: {in_t}, %o: {out_t}):
{body}
    }} -> tensor<{n}x{out_t}>
    return %r : tensor<{n}x{out_t}>
  }}
}}"""
    ensure_registered()
    res = lower_model(src, tmp_path / tag, targets=("host",), features=frozenset({EXACT_FEATURE}))
    assert "__truncsfbf2" not in res.ll_path.read_text()
    out = np.zeros(n, out_dtype)
    HostModel.load(str(res.host_so))([(inputs.ctypes.data, (n,)), (out.ctypes.data, (n,))])
    return out


@pytest.mark.skipif(not _host_toolchain(), reason="m2m venv / clang-23 missing")
def test_inline_bf16_rounding_is_the_runtime_helpers_on_every_class_of_input(tmp_path):
    """f32 -> bf16 rounds exactly as the runtime helper the default lowering calls -- ties to even on
    the bits, overflow to infinity, a NaN keeping its payload with the quiet bit set -- for every bf16
    value, every tie between two, either side of each tie, and random patterns."""
    import numpy as np

    every_bf16 = np.arange(1 << 16, dtype=np.uint32) << 16
    ties = every_bf16 | 0x8000
    noise = np.random.default_rng(0).integers(0, 1 << 32, 1 << 18, dtype=np.uint64).astype(np.uint32)
    bits = np.concatenate([every_bf16, ties, ties - 1, ties + 1, noise]).astype(np.uint32)
    body = "      %y = arith.truncf %x : f32 to bf16\n      linalg.yield %y : bf16"
    out = _run_unary(tmp_path, "trunc", body, "f32", "bf16", bits, np.uint16)
    assert np.array_equal(out, _truncsfbf2(bits))


@pytest.mark.skipif(not _host_toolchain(), reason="m2m venv / clang-23 missing")
def test_a_bf16_round_even_is_the_f32_one_rounded_for_every_bf16(tmp_path):
    import numpy as np

    every = np.arange(1 << 16, dtype=np.uint16)
    body = "      %y = math.roundeven %x : bf16\n      linalg.yield %y : bf16"
    out = _run_unary(tmp_path, "roundeven", body, "bf16", "bf16", every, np.uint16)
    widened = (every.astype(np.uint32) << 16).view(np.float32)
    finite = (every & 0x7F80) != 0x7F80
    want = _truncsfbf2(np.rint(widened[finite]).view(np.uint32))  # numpy rint rounds ties to even
    assert np.array_equal(out[finite], want)
    infinite = ~finite & ((every & 0x7F) == 0)
    assert np.array_equal(out[infinite], every[infinite])
    assert np.all((out[~finite & ~infinite] & 0x7F80) == 0x7F80) and np.all(out[~finite & ~infinite] & 0x7F)
