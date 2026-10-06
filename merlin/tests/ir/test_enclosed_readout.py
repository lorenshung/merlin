import ctypes
import subprocess

import pytest

from merlin.llvmlower.enclosed_readout import emit_pair_scan, prove
from merlin.llvmlower.requantization import quantized


@pytest.mark.parametrize("relu", [False, True])
@pytest.mark.parametrize("scales", [(0.5, 0.25), (0.125,), (1.0,)])
def test_complete_small_domains(scales, relu):
    product = 1.0
    for x in scales:
        product *= x
    p = prove(scales, [product, product], -4096, 4096, relu=relu)
    assert p["enclosed"] and p["exact_pair_decoder"]
    for x in range(-4096, 4097):
        assert quantized(x, scales, relu) == quantized(x, [product], relu)


def test_enclosure_does_not_imply_decodability():
    p = prove([0.25], [0.1, 0.4], -500, 500)
    assert p["enclosed"] and not p["exact_pair_decoder"]
    w = p["decoder_counterexamples"][0]
    assert w["first"]["source"] != w["other"]["source"]
    for x in (w["first"]["acc"], w["other"]["acc"]):
        assert [quantized(x, [s]) for s in p["store_scales"]] == w["pair"]
    with pytest.raises(ValueError):
        emit_pair_scan(p, "reject")


def test_non_enclosing_refused():
    p = prove([0.5], [1.0, 1.0], -100, 100)
    assert p["enclosure_counterexamples"] and not p["exact_pair_decoder"]
    with pytest.raises(ValueError):
        emit_pair_scan(p, "reject")


@pytest.mark.parametrize("scales", [[], [0.0, 1.0], [float("nan"), 1.0], [-1.0, 2.0]])
def test_invalid_scale_contract(scales):
    with pytest.raises(ValueError):
        prove([1.0], scales, -20, 20)


@pytest.mark.parametrize("copy_policy", ["runtime", "compiler_builtin"])
def test_packet_unaligned_tail_and_guards(tmp_path, copy_policy):
    # Independent synthetic exact correction: integer identity is between .99x
    # and 1.01x, and pairs identify all integer source values in this domain.
    p = prove([1.0], [0.99, 1.01], -90, 90)
    assert p["exact_pair_decoder"] and p["corrections"]
    source = tmp_path / "scan.c"
    source.write_text(emit_pair_scan(p, "scan", copy_policy=copy_policy))
    so = tmp_path / "scan.so"
    subprocess.run(["cc", "-shared", "-fPIC", "-O2", str(source), "-o", str(so)], check=True)
    f = ctypes.CDLL(str(so)).scan
    f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    for n in (0, 1, 7, 8, 9, 15, 16, 17, 181):
        aa = (ctypes.c_ubyte * (n + 5))(*([173] * (n + 5)))
        bb = (ctypes.c_ubyte * (n + 5))(*([219] * (n + 5)))
        values = list(range(-90, -90 + n))
        for j, x in enumerate(values):
            aa[j + 1] = quantized(x, [0.99]) & 255
            bb[j + 2] = quantized(x, [1.01]) & 255
        f(ctypes.byref(aa, 1), ctypes.byref(bb, 2), n)
        assert list(aa)[1 : n + 1] == [x & 255 for x in values]
        assert aa[0] == 173 and list(aa)[n + 1 :] == [173] * 4
    p["corrections"][0]["source"] += 1
    with pytest.raises(ValueError):
        emit_pair_scan(p, "scan")


def test_compiled_ordered_f32_oracle_and_conversion_edges(tmp_path):
    # Multiple rounded products, clipping, ties, signed int32->f32 conversion.
    cases = [
        ([0.5], [0.49, 0.51], -2048, 2048, False),
        ([0.001, 0.001], [0.00000099, 0.00000101], -(1 << 31), (1 << 31) - 1, False),
        ([0.125, 0.25], [0.03, 0.032], -10000, 10000, True),
    ]
    from merlin.llvmlower.integer_readout import derive, evaluate

    for idx, (scales, stores, lo, hi, relu) in enumerate(cases):
        source = derive(scales, lo, hi, relu)
        proof = prove(scales, stores, lo, hi, relu=relu)
        assert proof["enclosed"]
        code = "#include <math.h>\n#include <stdint.h>\nint f(int32_t a){volatile float x=(float)a;"
        code += "".join(f"x=x*{float(v).hex()}f;" for v in source["source_scales"])
        low = 0 if relu else -128
        code += f"x=nearbyintf(x);if(x<{low})x={low};if(x>127)x=127;return (int)x;}}"
        path = tmp_path / f"original{idx}.c"
        path.write_text(code)
        so = path.with_suffix(".so")
        subprocess.run(
            [
                "cc",
                "-shared",
                "-fPIC",
                "-O2",
                "-ffp-contract=off",
                "-frounding-math",
                "-fno-builtin",
                str(path),
                "-lm",
                "-o",
                str(so),
            ],
            check=True,
        )
        f = ctypes.CDLL(str(so)).f
        f.argtypes = [ctypes.c_int32]
        probes = {lo, hi, 0}
        for a in source["thresholds"] + [1 << 24, -(1 << 24)]:
            probes.update(x for x in [a - 1, a, a + 1] if lo <= x <= hi)
        for x in probes:
            assert f(x) == evaluate(x, source)


def test_fixed_source_ladder_and_refusal_contract():
    from merlin.llvmlower.enclosed_readout import synthesize

    p = synthesize([0.125, 0.25], -100000, 100000)
    assert p["accepted"] and p["radius"] == 0
    assert synthesize([1e-30, 1e-30], -100, 100)["accepted"] is False
    for radii in ((), [1], (True,), (-1,), (65537,)):
        with pytest.raises(ValueError):
            synthesize([0.5], -10, 10, radii=radii)


def test_decoder_compiles_freestanding_without_runtime_headers(tmp_path):
    import os

    clang = os.environ.get("MERLIN_CLANG")
    if not clang:
        pytest.skip("selected Clang required for freestanding build gate")
    p = prove([1.0], [0.99, 1.01], -90, 90)
    c = tmp_path / "scan.c"
    c.write_text(emit_pair_scan(p, "scan"))
    subprocess.run(
        [
            clang,
            "--target=riscv64-unknown-elf",
            "-march=rv64gc",
            "-ffreestanding",
            "-fno-builtin",
            "-O2",
            "-c",
            str(c),
            "-o",
            str(tmp_path / "scan.o"),
        ],
        check=True,
        capture_output=True,
    )


def test_compiler_builtin_copy_has_no_external_runtime_dependency(tmp_path):
    import os
    from pathlib import Path

    clang = os.environ.get("MERLIN_CLANG")
    if not clang:
        pytest.skip("selected compiler required")
    p = prove([1.0], [0.99, 1.01], -90, 90)
    c = tmp_path / "scan.c"
    c.write_text(emit_pair_scan(p, "scan", copy_policy="compiler_builtin"))
    obj = tmp_path / "scan.o"
    subprocess.run(
        [
            clang,
            "--target=riscv64-unknown-elf",
            "-march=rv64gc",
            "-ffreestanding",
            "-fno-builtin",
            "-O2",
            "-c",
            str(c),
            "-o",
            str(obj),
        ],
        check=True,
        capture_output=True,
    )
    result = subprocess.run(
        [str(Path(clang).with_name("llvm-nm")), "--undefined-only", str(obj)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert not result.stdout.strip()
    with pytest.raises(ValueError):
        emit_pair_scan(p, "scan", copy_policy="unknown")
