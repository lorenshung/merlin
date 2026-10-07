"""Source-DAG closure, placement-only refusal and actual special/FENV words."""

import ctypes
import hashlib
import subprocess

import numpy as np
import pytest

from merlin.llvmlower.source_continuation_outline import SourceContinuationBinding, outline_source_continuations
from merlin.llvmlower.source_expression_interval import IntervalEffectContract, ScalarExpression, ScalarStep
from merlin.llvmlower.toolchain import clang

EFFECTS = IntervalEffectContract(True, True, True, True, True)
EXPRESSION = ScalarExpression(
    (
        ScalarStep("arith.constant", (), "f32", 0x40000000),
        ScalarStep("arith.mulf", (0, 1), "f32"),
        ScalarStep("arith.constant", (), "f32", 0x40400000),
        ScalarStep("arith.divf", (2, 3), "f32"),
    )
)
SOURCE = """define float @continuation(float %x) alwaysinline {
  %a = fmul float %x, 2.000000e+00
  %b = fdiv float %a, 3.000000e+00
  ret float %b
}
"""


def bind(text, **kwargs):
    return outline_source_continuations(
        text,
        bindings=(SourceContinuationBinding("continuation", EXPRESSION),),
        expected_source_sha256=hashlib.sha256(text.encode()).hexdigest(),
        effects=EFFECTS,
        **kwargs,
    )


def test_default_identity_and_original_body_placement():
    assert outline_source_continuations("unknown empty policy")[0] == "unknown empty policy"
    changed, report = bind(SOURCE)
    assert changed == SOURCE.replace("alwaysinline", "noinline cold")
    assert report["routes"][0]["numeric_effect_alias_permissions_added"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s.replace("2.000000e+00", "2.000001e+00"),
        lambda s: s.replace("ret float", "%unused = fmul float %x, 2.000000e+00\n  ret float"),
        lambda s: s.replace("ret float", "store float %x, ptr @global\n  ret float"),
        lambda s: s.replace("ret float", "%opaque = call float @unknown(float %x)\n  ret float"),
        lambda s: s.replace("alwaysinline", "alwaysinline strictfp"),
        lambda s: s.replace("alwaysinline", "alwaysinline returns_twice"),
        lambda s: s.replace("fmul float", "fmul fast float"),
        lambda s: s.replace("float %x)", "float noundef %x)"),
    ],
)
def test_source_mutation_or_context_refuses_transactionally(mutation):
    text = mutation(SOURCE)
    before = text
    with pytest.raises(ValueError):
        bind(text)
    assert text == before


def test_wrong_witness_missing_effects_and_later_invalid_selection():
    with pytest.raises(ValueError, match="witness"):
        outline_source_continuations(
            SOURCE,
            bindings=(SourceContinuationBinding("continuation", EXPRESSION),),
            expected_source_sha256="0" * 64,
            effects=EFFECTS,
        )
    with pytest.raises(ValueError, match="effect"):
        outline_source_continuations(
            SOURCE,
            bindings=(SourceContinuationBinding("continuation", EXPRESSION),),
            expected_source_sha256=hashlib.sha256(SOURCE.encode()).hexdigest(),
        )
    text = SOURCE + SOURCE.replace("continuation", "second").replace("2.000000e+00", "4.000000e+00")
    with pytest.raises(ValueError, match="DAG"):
        outline_source_continuations(
            text,
            bindings=(
                SourceContinuationBinding("continuation", EXPRESSION),
                SourceContinuationBinding("second", EXPRESSION),
            ),
            expected_source_sha256=hashlib.sha256(text.encode()).hexdigest(),
            effects=EFFECTS,
        )
    assert "noinline" not in text


def test_actual_native_special_payloads_rounding_sticky_empty_tails(tmp_path):
    libraries = []
    for label, text in (("source", SOURCE), ("candidate", bind(SOURCE)[0])):
        path = tmp_path / (label + ".ll")
        path.write_text(text)
        library = tmp_path / (label + ".so")
        subprocess.run(
            [str(clang()), "-O3", "-ffp-contract=off", "-fPIC", "-shared", str(path), "-o", str(library)],
            check=True,
            capture_output=True,
        )
        lib = ctypes.CDLL(str(library))
        libraries.append(lib)
    source = tmp_path / "gate.c"
    source.write_text("""#include <fenv.h>
#include <stdint.h>
#include <string.h>
typedef float(*F)(float);
int check(F a,F b,const uint32_t*input,unsigned n,unsigned mode,unsigned preset){
 uint32_t ao[64],bo[64];fenv_t old;fegetenv(&old);
 const int modes[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};
 const int flags[]={0,FE_INVALID,FE_DIVBYZERO,FE_OVERFLOW,FE_UNDERFLOW,FE_INEXACT,FE_ALL_EXCEPT};
 memset(ao,73,sizeof(ao));memset(bo,73,sizeof(bo));fesetround(modes[mode]);
 feclearexcept(FE_ALL_EXCEPT);feraiseexcept(flags[preset]);
 for(unsigned i=0;i<n;i++){float x,y;memcpy(&x,input+i,4);y=a(x);memcpy(ao+8+i,&y,4);}
 int af=fetestexcept(FE_ALL_EXCEPT);
 feclearexcept(FE_ALL_EXCEPT);feraiseexcept(flags[preset]);
 for(unsigned i=0;i<n;i++){float x,y;memcpy(&x,input+i,4);y=b(x);memcpy(bo+8+i,&y,4);}
 int bf=fetestexcept(FE_ALL_EXCEPT);fesetenv(&old);
 return af!=bf?1:memcmp(ao,bo,sizeof(ao))?2:0;
}
""")
    library = tmp_path / "gate.so"
    subprocess.run(
        [str(clang()), "-O3", "-fno-builtin", "-fPIC", "-shared", str(source), "-lm", "-o", str(library)],
        check=True,
        capture_output=True,
    )
    lib = ctypes.CDLL(str(library))
    gate = lib.check
    gate.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_uint] * 3
    gate.restype = ctypes.c_int
    raw = np.array(
        [
            0,
            0x80000000,
            1,
            0x80000001,
            0x007FFFFF,
            0x00800000,
            0x7F7FFFFF,
            0xFF7FFFFF,
            0x7F800000,
            0xFF800000,
            0x7FC12345,
            0x7F812345,
            0xFF812345,
            0x3F800000,
            0xBF800000,
            0x3F000000,
            0x3F000001,
        ],
        np.uint32,
    )
    original = raw.copy()
    for n in (0, 1, 7, len(raw)):
        for mode in range(4):
            for preset in range(7):
                assert (
                    gate(
                        *(ctypes.cast(lib.continuation, ctypes.c_void_p) for lib in libraries),
                        raw.ctypes.data,
                        n,
                        mode,
                        preset,
                    )
                    == 0
                )
    np.testing.assert_array_equal(raw, original)
