"""Physical prepared rows preserve four independent source consumers."""

import ctypes as C
import importlib.util
import shutil
import subprocess
from dataclasses import replace

import numpy as np
import pytest

from merlin.common.paths import merlin_dir
from merlin.llvmlower.source_attention_frontier import emit_source_attention_frontier

_spec = importlib.util.spec_from_file_location(
    "frontier_fixture", merlin_dir() / "tests/runtime/test_source_attention_frontier.py"
)
fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixture)
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
        fixture.PLAN,
        replace(fixture.PLAN, heads=1, query_rows=1, depth=2, chunk=6, segment=2),
        replace(fixture.PLAN, heads=3, query_rows=5, depth=8, chunk=12, segment=5, denominator_lanes=4),
    ],
)
def native(tmp_path_factory, request):
    w = tmp_path_factory.mktemp("prepared-rhs")
    source = emit_source_attention_frontier(request.param, symbol="test_provider", prepare_readonly_rhs=True, **FLAGS)
    source += (
        fixture.EXTRA
        + """
int prepared_run(merlin_attention_view*in,merlin_attention_view*out,void*w,size_t cap,void*rhs,size_t bytes,void*epoch,size_t use){
return test_provider_with_rhs(in,out,w,cap,native_products,0,rhs,bytes,epoch,use);}
"""
    )
    (w / "probe.c").write_text(source)
    subprocess.run(
        [
            shutil.which("clang") or "cc",
            "-O2",
            "-shared",
            "-fPIC",
            "-ffp-contract=off",
            "-I",
            str(merlin_dir() / "runtime/c"),
            str(w / "probe.c"),
            "-lm",
            "-o",
            str(w / "probe.so"),
        ],
        check=True,
    )
    lib = C.CDLL(str(w / "probe.so"))
    lib.test_provider_workspace_bytes.restype = lib.test_provider_rhs_owner_bytes.restype = C.c_size_t
    lib.test_provider_rhs_owner_invalidate.argtypes = [C.c_void_p, C.c_size_t]
    lib.test_provider_rhs_prepare.argtypes = [C.POINTER(fixture.View), C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t]
    lib.prepared_run.argtypes = [
        C.POINTER(fixture.View),
        C.POINTER(fixture.View),
        C.c_void_p,
        C.c_size_t,
        C.c_void_p,
        C.c_size_t,
        C.c_void_p,
        C.c_size_t,
    ]
    lib.oracle.argtypes = [C.POINTER(fixture.View), C.POINTER(C.c_uint16)]
    lib.plan = request.param
    return lib


def inputs(plan, seed, masked=False, strided=False):
    rng = np.random.default_rng(seed)
    arrays = []
    for i in range(11):
        if i in (2, 7):
            value = rng.integers(0, 2, (1, 1, plan.query_rows, plan.chunk), dtype=np.uint8)
            if masked:
                value[:] = 0
        else:
            rows = (
                plan.query_rows
                if i == 0
                else plan.chunk
                if i in (1, 6)
                else min(plan.segment, plan.chunk - ((i - 3) if i < 6 else (i - 8)) * plan.segment)
            )
            floats = rng.integers(-8, 9, (1, plan.heads, rows, plan.depth)).astype(np.float32) / 8
            value = (floats.view(np.uint32) >> 16).astype(np.uint16)
        if strided:
            backing = np.zeros((*value.shape[:-1], 2 * value.shape[-1]), dtype=value.dtype)
            backing[..., ::2] = value
            value = backing[..., ::2]
        arrays.append(value)
    return arrays


def quant(words, plan):
    floats = (words.astype(np.uint32) << 16).view(np.float32)[0].transpose(1, 0, 2).reshape(plan.query_rows, -1)

    def bf(x):
        bits = np.ascontiguousarray(x, np.float32).view(np.uint32)
        return ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000).view(np.float32)

    scale = np.maximum(bf(abs(floats).max(1) / np.float32(plan.quant_divisor)), np.float32(plan.quant_epsilon))
    inverse = bf(np.float32(1) / scale)
    values = np.clip(bf(np.rint(bf(floats * inverse[:, None]))), plan.quant_lower, plan.quant_upper).astype(np.int8)
    return values, scale.view(np.uint32)


def allocation(size):
    raw = np.full(size + 128, 0xAC, np.uint8)
    address = (raw.ctypes.data + 63) & -64
    return raw, address


@pytest.mark.parametrize("seed", [0, 3, 19])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_four_consumers_exact_original_quant_and_stale_epoch(native, seed, masked, strided):
    xs = inputs(native.plan, seed, masked=masked, strided=strided)
    original = [a.copy() for a in xs]
    views = (fixture.View * 11)(*[fixture.view(a) for a in xs])
    size, rhs_size = native.test_provider_workspace_bytes(), native.test_provider_rhs_owner_bytes()
    work, wp = allocation(size)
    rhs, rp = allocation(rhs_size)
    epoch = C.c_int(1)
    assert native.test_provider_rhs_prepare(views, rp, rhs_size, C.byref(epoch), 4) == 1
    for use in range(4):
        out = np.full((1, native.plan.heads, native.plan.query_rows, native.plan.depth), 0xDEAD, np.uint16)
        oracle = np.empty_like(out)
        ov = fixture.view(out)
        assert native.prepared_run(views, C.byref(ov), wp, size, rp, rhs_size, C.byref(epoch), use) == 1
        assert native.oracle(views, oracle.ctypes.data_as(C.POINTER(C.c_uint16))) == 1
        assert all(
            np.array_equal(a, b) for a, b in zip(quant(out, native.plan), quant(oracle, native.plan), strict=True)
        )
    assert native.prepared_run(views, C.byref(ov), wp, size, rp, rhs_size, C.byref(epoch), 4) == 0
    assert all(np.array_equal(a, b) for a, b in zip(xs, original, strict=True))
    for raw, pointer, extent in [(work, wp, size), (rhs, rp, rhs_size)]:
        off = pointer - raw.ctypes.data
        assert (raw[:off] == 0xAC).all() and (raw[off + extent :] == 0xAC).all()


def test_owner_alias_and_unknown_epoch_refuse(native):
    xs = inputs(native.plan, 2)
    views = (fixture.View * 11)(*[fixture.view(a) for a in xs])
    size = native.test_provider_rhs_owner_bytes()
    raw, pointer = allocation(size)
    epoch = C.c_int(7)
    assert native.test_provider_rhs_prepare(views, pointer, size - 1, C.byref(epoch), 4) == 0
    assert native.test_provider_rhs_prepare(views, pointer, size, None, 4) == 0
    # An input view inside private storage is forbidden before any preparation writes.
    views[0].data = pointer
    assert native.test_provider_rhs_prepare(views, pointer, size, C.byref(epoch), 4) == 0
    assert (raw == 0xAC).all()


def test_default_identity_and_explicit_selection_refusal():
    assert emit_source_attention_frontier(fixture.PLAN, symbol="x", **FLAGS) == emit_source_attention_frontier(
        fixture.PLAN, symbol="x", prepare_readonly_rhs=False, **FLAGS
    )
    for value in (1, None, "yes"):
        with pytest.raises(ValueError):
            emit_source_attention_frontier(fixture.PLAN, symbol="x", prepare_readonly_rhs=value, **FLAGS)


def test_nonfinite_source_refuses_before_any_consumer(native):
    xs = inputs(native.plan, 6)
    xs[1].flat[0] = 0x7FC1
    views = (fixture.View * 11)(*[fixture.view(a) for a in xs])
    size = native.test_provider_rhs_owner_bytes()
    raw, pointer = allocation(size)
    epoch = C.c_int(2)
    assert native.test_provider_rhs_prepare(views, pointer, size, C.byref(epoch), 4) == 0


def test_prepare_bridge_can_invalidate_before_descriptor_refusal(native):
    xs = inputs(native.plan, 9)
    original = [a.copy() for a in xs]
    views = (fixture.View * 11)(*[fixture.view(a) for a in xs])
    size, rhs_size = native.test_provider_workspace_bytes(), native.test_provider_rhs_owner_bytes()
    work, wp = allocation(size)
    rhs, rp = allocation(rhs_size)
    epoch = C.c_int(3)
    assert native.test_provider_rhs_prepare(views, rp, rhs_size, C.byref(epoch), 4) == 1
    assert native.test_provider_rhs_owner_invalidate(None, rhs_size) == 0
    assert native.test_provider_rhs_owner_invalidate(rp, rhs_size - 1) == 0
    assert native.test_provider_rhs_owner_invalidate(rp, rhs_size) == 1
    # A failed subsequent prepare may return before touching validity.
    assert native.test_provider_rhs_prepare(views, rp, rhs_size, None, 4) == 0
    out = np.full((1, native.plan.heads, native.plan.query_rows, native.plan.depth), 0xDEAD, np.uint16)
    ov = fixture.view(out)
    assert native.prepared_run(views, C.byref(ov), wp, size, rp, rhs_size, C.byref(epoch), 0) == 0
    assert (out == 0xDEAD).all()
    assert all(np.array_equal(a, b) for a, b in zip(xs, original, strict=True))
