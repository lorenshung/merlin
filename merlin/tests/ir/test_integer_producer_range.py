"""Producer-bound readout removes only a mathematically redundant domain trap."""

import ctypes
import itertools
import subprocess

import numpy as np
import pytest

from merlin.llvmlower.integer_producer_range import IntegerSumProductsRange as Domain
from merlin.llvmlower.integer_readout import derive, emit_readout


def test_prefix_enclosure_including_bias_and_negative_products():
    domain = Domain(3, -2, 1, -1, 2, -3, 4)
    assert domain.interval() == (-15, 10)
    for seed in range(-3, 5):
        for terms in itertools.product(sorted({a * b for a in range(-2, 2) for b in range(-1, 3)}), repeat=3):
            # Every admitted product is enclosed, including correlated operands.
            value = seed
            for term in terms:
                value += term
                assert -15 <= value <= 10


def test_minus128_and_zero_terms():
    assert Domain(2304, -128, 127, -128, 127).interval() == (-37453824, 37748736)
    assert Domain(0, -128, 127, -128, 127, -3, 5).interval() == (-3, 5)


@pytest.mark.parametrize(
    "domain",
    [
        Domain(True, -1, 1, -1, 1),
        Domain(-1, -1, 1, -1, 1),
        Domain(1, 2, 1, -1, 1),
        Domain(1, -1, 1, -1, 1, 2, 1),
        Domain(1, 1 << 30, 1 << 30, 2, 2),
        Domain(200000, -128, 127, -128, 127),
        Domain(2, -1, -1, 1, 1, -(1 << 31), -(1 << 31)),
    ],
)
def test_refuses_malformed_or_overflowing_producers(domain):
    with pytest.raises(ValueError):
        domain.interval()


def test_requires_typed_contained_producer():
    proof = derive([0.25], -10, 10)
    for domain in ({"terms": 1}, True, Domain(1, -128, 127, -128, 127)):
        with pytest.raises(ValueError):
            emit_readout(proof, "bad", producer_range=domain)


def test_compiled_original_rounding_all_domain_and_packet_tails(tmp_path):
    domain = Domain(3, -16, 15, -16, 15, -5, 7)
    lo, hi = domain.interval()
    proof = derive([0.125, 0.75], lo, hi)
    code = emit_readout(proof, "checked", fixedpoint=True, saturation_first=True, packet=8)
    code += emit_readout(proof, "bound", fixedpoint=True, saturation_first=True, packet=8, producer_range=domain)
    code += """\n#include <math.h>
void original(const int32_t*a,int8_t*b,size_t n){for(size_t i=0;i<n;i++){
volatile float y=(float)a[i];y=y*0.125f;y=y*0.75f;
float z=nearbyintf(y);if(z<-128)z=-128;if(z>127)z=127;b[i]=(int8_t)z;}}
"""
    src = tmp_path / "readout.c"
    src.write_text(code)
    so = tmp_path / "readout.so"
    subprocess.run(["cc", "-O2", "-ffp-contract=off", "-shared", "-fPIC", str(src), "-lm", "-o", str(so)], check=True)
    lib = ctypes.CDLL(str(so))
    a = np.arange(lo, hi + 1, dtype=np.int32)
    for n in [0, 1, 7, 8, 9, a.size]:
        outputs = []
        for name in ("original", "checked", "bound"):
            f = getattr(lib, name)
            f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
            b = np.full(n + 17, -91, np.int8)
            f(a.ctypes.data, b.ctypes.data, n)
            assert np.all(b[n:] == -91)
            outputs.append(b)
        assert all(np.array_equal(outputs[0], b) for b in outputs[1:])
