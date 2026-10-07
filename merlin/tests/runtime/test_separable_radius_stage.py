"""Independent radius columns preserve scalar bits, flags and refusals."""

import ctypes as C
import shutil
import subprocess

import pytest
from test_separable_fma_radius import SOURCE, test_original_ordered_fma_and_fallback

from merlin.common.paths import merlin_dir

CHECK = r"""
  if(plan.valid&&n>=8){
   float staged_lo[8],staged_hi[8],scalar_lo[8],scalar_hi[8];
   for(int first=0;first+8<=n;first+=8){
    feclearexcept(FE_ALL_EXCEPT);feraiseexcept(FE_INVALID);
    for(int c=0;c<8;c++)if(!merlin_fma_separable_radius_apply(&plan,first+c,&scalar_lo[c],&scalar_hi[c]))return -10;
    int flags=fetestexcept(FE_ALL_EXCEPT);
    feclearexcept(FE_ALL_EXCEPT);feraiseexcept(FE_INVALID);
    if(!merlin_fma_separable_radius_apply_eight(&plan,first,staged_lo,staged_hi))return -11;
    if(flags!=fetestexcept(FE_ALL_EXCEPT)||memcmp(staged_lo,scalar_lo,sizeof(staged_lo))||memcmp(staged_hi,scalar_hi,sizeof(staged_hi)))return -12;
   }
   if(merlin_fma_separable_radius_apply_eight(&plan,n-7,staged_lo,staged_hi))return -13;
   if(merlin_fma_separable_radius_apply_eight(&plan,0,staged_lo,staged_lo))return -14;
   if(merlin_fma_separable_radius_apply_eight(&plan,0,(float*)plan.centers,staged_hi))return -15;
   if(merlin_fma_separable_radius_apply_eight(&plan,0,(float*)plan.columns->source,staged_hi))return -16;
   if(merlin_fma_separable_radius_apply_eight(0,0,staged_lo,staged_hi))return -17;
  }
"""
SOURCE = "#include <fenv.h>\n" + SOURCE.replace("*admissions+=plan.valid;", "*admissions+=plan.valid;" + CHECK)


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    cc = shutil.which("cc")
    if not cc:
        pytest.skip("native C compiler required")
    out = tmp_path_factory.mktemp("radius-stage")
    source = out / "probe.c"
    source.write_text(SOURCE)
    library = out / "probe.so"
    subprocess.run(
        [
            cc,
            "-O2",
            "-frounding-math",
            "-ffp-contract=off",
            "-shared",
            "-fPIC",
            "-I",
            str(merlin_dir() / "runtime/c"),
            str(source),
            "-lm",
            "-o",
            str(library),
        ],
        check=True,
    )
    result = C.CDLL(str(library))
    result.compare.argtypes = [C.c_void_p] * 4 + [C.c_int] * 4 + [C.POINTER(C.c_int)]
    return result


@pytest.mark.parametrize("shape", [(1, 8, 1), (3, 9, 7), (2, 16, 31), (2, 17, 31)])
@pytest.mark.parametrize("case", ["zero", "exact", "representation", "uncertain", "overflow", "subnormal", "cancel"])
def test_stage_original_source_bits_flags_and_alias_refusal(lib, shape, case):
    test_original_ordered_fma_and_fallback(lib, shape, case)


@pytest.mark.parametrize("dimensions", [(2, 3, 4, 8, 3, 2), (1, 5, 8, 12, 5, 4)])
def test_regular_emitter_complete_source_observer_and_dirty_guards(tmp_path, dimensions):
    from dataclasses import replace

    import numpy as np
    from test_prepared_attention_rhs import inputs, quant
    from test_source_attention_frontier import EXTRA, PLAN, View, view

    from merlin.llvmlower.exact_bound_conversion import ExactBoundConversionContract
    from merlin.llvmlower.independent_lane_schedule import LaneEffects
    from merlin.llvmlower.source_attention_frontier import emit_source_attention_frontier

    h, m, d, k, segment, lanes = dimensions
    plan = replace(PLAN, heads=h, query_rows=m, depth=d, chunk=k, segment=segment, denominator_lanes=lanes)
    prefix = '#include "ordered_fma_bounds.h"\nstatic float narrow_lo(double x){float f=(float)x;return (double)f>x?merlin_fma_next_down_f32(f):f;}\nstatic float narrow_hi(double x){float f=(float)x;return (double)f<x?merlin_fma_next_up_f32(f):f;}\n#define MERLIN_F32_EXACT_FLOOR_FROM_F64(x) narrow_lo(x)\n#define MERLIN_F32_EXACT_CEIL_FROM_F64(x) narrow_hi(x)\n'
    source = (
        prefix
        + emit_source_attention_frontier(
            plan,
            symbol="test_provider",
            prepare_product_domain=True,
            prepare_required_norms=True,
            separable_source_radius=True,
            exact_bound_conversion=ExactBoundConversionContract(True, True, True, True, True),
            radius_stage_effects=LaneEffects(True, True, True, True),
        )
        + EXTRA
    )
    import hashlib

    library = tmp_path / ("p_" + hashlib.sha256(source.encode()).hexdigest() + ".so")
    (tmp_path / "p.c").write_text(source)
    subprocess.run(
        [
            shutil.which("cc"),
            "-O2",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-shared",
            "-fPIC",
            "-I",
            str(merlin_dir() / "runtime/c"),
            str(tmp_path / "p.c"),
            "-lm",
            "-o",
            str(library),
        ],
        check=True,
    )
    lib = C.CDLL(str(library))
    lib.test_provider_workspace_bytes.restype = C.c_size_t
    lib.run.argtypes = [C.POINTER(View), C.POINTER(View), C.c_void_p, C.c_size_t, C.c_int]
    lib.oracle.argtypes = [C.POINTER(View), C.c_void_p]
    for masked in (False, True):
        xs = inputs(plan, seed=78, masked=masked, strided=True)
        original = [x.copy() for x in xs]
        views = (View * 11)(*[view(x) for x in xs])
        size = lib.test_provider_workspace_bytes()
        scratch = np.full(size + 64, 0xAC, np.uint8)
        pointer = (scratch.ctypes.data + 7) & -8
        out = np.full((1, h, m, d), 0xDEAD, np.uint16)
        gold = np.empty_like(out)
        descriptor = view(out)
        assert lib.run(views, C.byref(descriptor), pointer, size, 0) == 1
        assert lib.oracle(views, gold.ctypes.data) == 1
        assert all(np.array_equal(a, b) for a, b in zip(quant(out, plan), quant(gold, plan), strict=True))
        assert all(np.array_equal(a, b) for a, b in zip(xs, original, strict=True))
        offset = pointer - scratch.ctypes.data
        assert (scratch[:offset] == 0xAC).all() and (scratch[offset + size :] == 0xAC).all()
        out.fill(0xDEAD)
        assert lib.run(views, C.byref(descriptor), pointer, size, 1) == 0
        assert (out == 0xDEAD).all()
