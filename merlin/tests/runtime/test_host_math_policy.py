"""Explicit precision policy changes only selected libm calls and linked identity."""

import ctypes
import decimal
import math
import shutil
import struct
import subprocess

import pytest

from merlin.runtime.backends.spike_model import _supplemental_object_digest
from merlin.runtime.host_math import build_host_math


def test_default_is_emission_neutral_without_compiler(tmp_path):
    def forbidden(_):
        raise AssertionError("default must not invoke a compiler")

    work = tmp_path / "absent"
    assert build_host_math("native", work, "absent-compiler", [], forbidden) == ([], ())
    assert not work.exists()


@pytest.mark.parametrize("policy", [None, True, "double", "expf_via_double "])
def test_unknown_policy_refuses_before_io(tmp_path, policy):
    with pytest.raises(ValueError, match="host_math_policy"):
        build_host_math(policy, tmp_path / "absent", "absent", [], lambda _: None)
    assert not (tmp_path / "absent").exists()


def test_missing_compiler_output_refuses(tmp_path):
    with pytest.raises(RuntimeError, match="did not produce"):
        build_host_math("expf_via_double", tmp_path, "absent", [], lambda _: None)


def test_real_link_selection_special_values_and_precision(tmp_path):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("requires C compiler and libm")

    def run(cmd):
        return subprocess.run(list(map(str, cmd)), check=True, capture_output=True)

    # A deliberately distinguishable original implementation proves linker interception.
    src = tmp_path / "caller.c"
    src.write_text("extern float expf(float); float evaluate(float x){return expf(x);}\n")
    stub = tmp_path / "native.c"
    stub.write_text("float expf(float x){return x-17.0f;}\n")
    for source in (src, stub):
        run([cc, "-O2", "-fPIC", "-fno-builtin", "-c", source, "-o", source.with_suffix(".o")])
    common = [src.with_suffix(".o"), stub.with_suffix(".o")]
    native, native_flags = build_host_math("native", tmp_path / "native", cc, ["-O2", "-fPIC"], run)
    selected, flags = build_host_math("expf_via_double", tmp_path / "selected", cc, ["-O2", "-fPIC"], run)
    for name, objects, extra in [("baseline", [], ()), ("native", native, native_flags), ("selected", selected, flags)]:
        run([cc, "-shared", "-Wl,-Bsymbolic", *common, *objects, *extra, "-lm", "-o", tmp_path / (name + ".so")])
    assert (tmp_path / "baseline.so").read_bytes() == (tmp_path / "native.so").read_bytes()
    assert _supplemental_object_digest(native) != _supplemental_object_digest(selected)
    libs = [ctypes.CDLL(str(tmp_path / (n + ".so"))) for n in ("native", "selected")]
    for lib in libs:
        lib.evaluate.argtypes = [ctypes.c_float]
        lib.evaluate.restype = ctypes.c_float
    assert libs[0].evaluate(0) == -17
    assert libs[1].evaluate(0) == 1
    assert libs[1].evaluate(-0.0) == 1
    assert libs[1].evaluate(float("-inf")) == 0
    assert libs[1].evaluate(float("inf")) == float("inf")
    assert math.isnan(libs[1].evaluate(float("nan")))

    def bits(x):
        return struct.pack("<f", x)

    for value in [-100.0, -90.0, -16.125, -1.0, -0.03125, 0.5, 1.0, 10.125, 80.0, 88.0]:
        x = ctypes.c_float(value).value
        with decimal.localcontext() as context:
            context.prec = 100
            expected = ctypes.c_float(float(decimal.Decimal(x).exp())).value
        assert bits(libs[1].evaluate(x)) == bits(expected)
