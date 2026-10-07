"""Normal explicit estimate execution and retained source refusal behavior.

Passing these finite fixtures does not turn a statistical model into a theorem.
"""

import ctypes as C
import hashlib
import shutil
import subprocess
from dataclasses import replace

import numpy as np
import pytest
from test_prepared_attention_rhs import inputs
from test_source_attention_frontier import EXTRA, PLAN, View, view

from merlin.common.paths import merlin_dir
from merlin.llvmlower.source_attention_frontier import emit_source_attention_frontier
from merlin.llvmlower.source_roundoff_policy import ApproximateSourceRoundoffPolicy

POLICY = ApproximateSourceRoundoffPolicy(True, True, True, True, True, True, True, True)
FLAGS = dict(
    prepare_product_domain=True,
    prepare_required_norms=True,
    separable_source_radius=True,
    prepare_softmax_domain=True,
    prepare_probability_bins=True,
    prepare_encoded_rows=True,
    prepare_softmax_spans=True,
    prepare_probability_points=True,
)


@pytest.fixture(
    scope="module",
    params=[
        PLAN,
        replace(PLAN, heads=1, query_rows=5, depth=16, chunk=48, segment=16, denominator_lanes=8, score_scale=0.0625),
    ],
)
def lib(tmp_path_factory, request):
    root = tmp_path_factory.mktemp("normal-rms4")
    source = (
        emit_source_attention_frontier(request.param, symbol="test_provider", source_roundoff_estimate=POLICY, **FLAGS)
        + EXTRA
    )
    source += "\nint set_round(int n){int modes[4]={FE_TONEAREST,FE_UPWARD,FE_DOWNWARD,FE_TOWARDZERO};return fesetround(modes[n]);}\n"
    path = root / ("provider_" + hashlib.sha256(source.encode()).hexdigest() + ".so")
    (root / "provider.c").write_text(source)
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
            str(root / "provider.c"),
            "-lm",
            "-o",
            str(path),
        ],
        check=True,
    )
    result = C.CDLL(str(path))
    result.test_provider_workspace_bytes.restype = C.c_size_t
    result.run.argtypes = [C.POINTER(View), C.POINTER(View), C.c_void_p, C.c_size_t, C.c_int]
    result.oracle.argtypes = [C.POINTER(View), C.c_void_p]
    result.plan = request.param
    return result


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_normal_finite_original_output_gate_and_ownership(lib, masked, strided):
    plan = lib.plan
    xs = inputs(plan, 39, masked=masked, strided=strided)
    old = [x.copy() for x in xs]
    descriptors = (View * 11)(*[view(x) for x in xs])
    out = np.full((1, plan.heads, plan.query_rows, plan.depth), 0xDEAD, np.uint16)
    reference = np.empty_like(out)
    descriptor = view(out)
    original_descriptor = bytes(descriptor)
    size = lib.test_provider_workspace_bytes()
    raw = np.full(size + 64, 0xCD, np.uint8)
    pointer = (raw.ctypes.data + 7) & -8
    assert lib.run(descriptors, C.byref(descriptor), pointer, size, 0) == 1
    assert lib.oracle(descriptors, reference.ctypes.data) == 1
    # Explicit finite fixture gate only, not an all-input accuracy certificate.
    actual = (out.astype(np.uint32) << 16).view(np.float32)
    source = (reference.astype(np.uint32) << 16).view(np.float32)
    assert np.allclose(actual, source, atol=0.03125, rtol=0.02)
    assert bytes(descriptor) == original_descriptor
    assert all(np.array_equal(a, b) for a, b in zip(xs, old, strict=True))
    offset = pointer - raw.ctypes.data
    assert (raw[:offset] == 0xCD).all() and (raw[offset + size :] == 0xCD).all()


@pytest.mark.parametrize("rounding", [1, 2, 3])
def test_non_rne_refuses_without_publication_then_original_source_replays(lib, rounding):
    plan = lib.plan
    xs = inputs(plan, 45)
    descriptors = (View * 11)(*[view(x) for x in xs])
    out = np.full((1, plan.heads, plan.query_rows, plan.depth), 0xDEAD, np.uint16)
    descriptor = view(out)
    size = lib.test_provider_workspace_bytes()
    raw = np.zeros(size + 64, np.uint8)
    pointer = (raw.ctypes.data + 7) & -8
    assert lib.set_round(rounding) == 0
    try:
        assert lib.run(descriptors, C.byref(descriptor), pointer, size, 0) == 0
        assert (out == 0xDEAD).all()
        # The unchanged source continuation is the required wrapper behavior.
        assert lib.oracle(descriptors, out.ctypes.data) == 1
    finally:
        assert lib.set_round(0) == 0
    assert not (out == 0xDEAD).all()


def test_failed_device_product_and_nonfinite_source_refuse(lib):
    plan = lib.plan
    xs = inputs(plan, 17)
    views = (View * 11)(*[view(x) for x in xs])
    size = lib.test_provider_workspace_bytes()
    raw = np.zeros(size + 64, np.uint8)
    pointer = (raw.ctypes.data + 7) & -8
    out = np.full((1, plan.heads, plan.query_rows, plan.depth), 0xDEAD, np.uint16)
    descriptor = view(out)
    assert lib.run(views, C.byref(descriptor), pointer, size, 1) == 0
    assert (out == 0xDEAD).all()
    xs[0].flat[0] = 0x7FC1
    assert lib.run(views, C.byref(descriptor), pointer, size, 0) == 0
    assert (out == 0xDEAD).all()


def test_normal_rms_schedule_and_prepared_owner_compose(tmp_path):
    from test_prepared_attention_rhs import allocation

    from merlin.llvmlower.exact_bound_conversion import ExactBoundConversionContract
    from merlin.llvmlower.independent_lane_schedule import LaneEffects

    prefix = '#include "ordered_fma_bounds.h"\nstatic float narrow_lo(double x){float f=(float)x;return (double)f>x?merlin_fma_next_down_f32(f):f;}\nstatic float narrow_hi(double x){float f=(float)x;return (double)f<x?merlin_fma_next_up_f32(f):f;}\n#define MERLIN_F32_EXACT_FLOOR_FROM_F64(x) narrow_lo(x)\n#define MERLIN_F32_EXACT_CEIL_FROM_F64(x) narrow_hi(x)\n'
    plan = replace(PLAN, heads=1, query_rows=5, depth=16, chunk=48, segment=16, denominator_lanes=8)
    source = (
        prefix
        + emit_source_attention_frontier(
            plan,
            symbol="test_provider",
            source_roundoff_estimate=POLICY,
            prepare_readonly_rhs=True,
            radius_stage_effects=LaneEffects(True, True, True, True),
            exact_bound_conversion=ExactBoundConversionContract(True, True, True, True, True),
            **FLAGS,
        )
        + EXTRA
        + "\nint prepared_run(merlin_attention_view*in,merlin_attention_view*out,void*w,size_t cap,void*rhs,size_t bytes,void*epoch,size_t use){return test_provider_with_rhs(in,out,w,cap,native_products,0,rhs,bytes,epoch,use);}\n"
    )
    library = tmp_path / ("composed_" + hashlib.sha256(source.encode()).hexdigest() + ".so")
    path = tmp_path / "composed.c"
    path.write_text(source)
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
            str(path),
            "-lm",
            "-o",
            str(library),
        ],
        check=True,
    )
    lib = C.CDLL(str(library))
    lib.test_provider_workspace_bytes.restype = lib.test_provider_rhs_owner_bytes.restype = C.c_size_t
    lib.test_provider_rhs_prepare.argtypes = [C.POINTER(View), C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t]
    lib.prepared_run.argtypes = [
        C.POINTER(View),
        C.POINTER(View),
        C.c_void_p,
        C.c_size_t,
        C.c_void_p,
        C.c_size_t,
        C.c_void_p,
        C.c_size_t,
    ]
    lib.oracle.argtypes = [C.POINTER(View), C.c_void_p]
    xs = inputs(plan, 781, strided=True)
    originals = [x.copy() for x in xs]
    descriptors = (View * 11)(*[view(x) for x in xs])
    size, rhs_size = lib.test_provider_workspace_bytes(), lib.test_provider_rhs_owner_bytes()
    work, wp = allocation(size)
    rhs, rp = allocation(rhs_size)
    epoch = C.c_int(7)
    assert lib.test_provider_rhs_prepare(descriptors, rp, rhs_size, C.byref(epoch), 2) == 1
    for use in range(2):
        out = np.full((1, plan.heads, plan.query_rows, plan.depth), 0xDEAD, np.uint16)
        original = np.empty_like(out)
        descriptor = view(out)
        descriptor_bytes = bytes(descriptor)
        assert lib.prepared_run(descriptors, C.byref(descriptor), wp, size, rp, rhs_size, C.byref(epoch), use) == 1
        assert lib.oracle(descriptors, original.ctypes.data) == 1
        actual_f32 = (out.astype(np.uint32) << 16).view(np.float32)
        source_f32 = (original.astype(np.uint32) << 16).view(np.float32)
        assert np.allclose(actual_f32, source_f32, atol=0.03125, rtol=0.02)
        assert bytes(descriptor) == descriptor_bytes
    assert lib.prepared_run(descriptors, C.byref(descriptor), wp, size, rp, rhs_size, C.byref(epoch), 2) == 0
    assert all(np.array_equal(a, b) for a, b in zip(xs, originals, strict=True))
    for raw, address, extent in ((work, wp, size), (rhs, rp, rhs_size)):
        offset = address - raw.ctypes.data
        assert (raw[:offset] == 0xAC).all() and (raw[offset + extent :] == 0xAC).all()
