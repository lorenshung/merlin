"""Complete source products, prepared borrows, original observations and refusal."""

import ctypes as C
import importlib.util
import subprocess
from dataclasses import replace

import numpy as np
import pytest

from merlin.common.paths import data_path, merlin_dir
from merlin.llvmlower.source_attention_frontier import emit_source_attention_frontier
from merlin.llvmlower.source_product_family import SourceProductFamilyContract

_spec = importlib.util.spec_from_file_location(
    "family_rhs_fixture", merlin_dir() / "tests/runtime/test_prepared_attention_rhs.py"
)
rhs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rhs)
fixture = rhs.fixture
CONTRACT = SourceProductFamilyContract(*([True] * 6))


@pytest.fixture(
    scope="module",
    params=[
        (p, prepared)
        for p in (
            fixture.PLAN,
            replace(fixture.PLAN, heads=3, query_rows=5, depth=8, chunk=12, segment=5, denominator_lanes=4),
        )
        for prepared in (False, True)
    ],
)
def pair(tmp_path_factory, request):
    plan, prepared = request.param
    work = tmp_path_factory.mktemp("complete-source-family")
    libraries = []
    for selected in (False, True):
        path = work / ("family" if selected else "control")
        source = emit_source_attention_frontier(
            plan,
            symbol="test_provider",
            integer_reconstruction=True,
            fuse_integer_reconstruction=True,
            prepare_readonly_rhs=prepared,
            source_product_family=CONTRACT if selected else None,
            **(rhs.FLAGS if prepared else {}),
        )
        extra = fixture.EXTRA
        if selected:
            start, finish = extra.index("static int native_products("), extra.index("\nint run(")
            extra = (
                extra[:start]
                + r"""
static int native_products(void*opaque,const int8_t*a,const int8_t*b,int32_t*c,int m,int n,int k,size_t stride){
 product_calls++;if(opaque==(void*)1)return 0;
 if(stride!=ROWS*CHUNK)return 0;
 for(int degree=0;degree<5;degree++)for(int row=0;row<m;row++)for(int col=0;col<n;col++){
  int64_t sum=0;for(int ad=0;ad<3;ad++){int bd=degree-ad;if(bd<0||bd>2)continue;
   for(int z=0;z<k;z++)sum+=(int32_t)a[ad*m*k+row*k+z]*(int32_t)b[bd*k*n+z*n+col];}
  c[degree*stride+row*n+col]=(int32_t)sum;
  if(opaque==(void*)2)return 0;
 }return 1;
}
"""
                + extra[finish:]
            )
        else:
            extra = extra.replace(" if(opaque)return 0;", " product_calls++;if(opaque)return 0;", 1)
        source += "\nstatic size_t product_calls;size_t get_product_calls(void){return product_calls;}\n" + extra
        if prepared:
            source += (
                "\nint prepared_run(merlin_attention_view*in,merlin_attention_view*out,void*w,size_t cap,"
                "void*rhs,size_t bytes,void*epoch,size_t use){return test_provider_with_rhs("
                "in,out,w,cap,native_products,0,rhs,bytes,epoch,use);}\n"
            )
        path.with_suffix(".c").write_text(source)
        subprocess.run(
            [
                "/usr/bin/cc",
                "-O2",
                "-fno-fast-math",
                "-ffp-contract=off",
                "-shared",
                "-fPIC",
                "-I",
                str(data_path("runtime", "c")),
                str(path.with_suffix(".c")),
                "-lm",
                "-o",
                str(path.with_suffix(".so")),
            ],
            check=True,
        )
        lib = C.CDLL(str(path.with_suffix(".so")))
        lib.test_provider_workspace_bytes.restype = lib.get_product_calls.restype = C.c_size_t
        lib.run.argtypes = [C.POINTER(fixture.View), C.POINTER(fixture.View), C.c_void_p, C.c_size_t, C.c_int]
        if prepared:
            lib.test_provider_rhs_owner_bytes.restype = C.c_size_t
            lib.test_provider_rhs_prepare.argtypes = [
                C.POINTER(fixture.View),
                C.c_void_p,
                C.c_size_t,
                C.c_void_p,
                C.c_size_t,
            ]
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
        libraries.append(lib)
    return plan, prepared, libraries


@pytest.mark.parametrize("seed", [3, 19])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_all_planes_source_observations_lifetime_and_calls(pair, seed, masked, strided):
    plan, prepared, libraries = pair
    xs = rhs.inputs(plan, seed, masked=masked, strided=strided)
    originals = [x.copy() for x in xs]
    views = (fixture.View * 11)(*[fixture.view(x) for x in xs])
    results, callback_counts = [], []
    for lib in libraries:
        size = lib.test_provider_workspace_bytes()
        raw, pointer = rhs.allocation(size)
        owner = None
        epoch = C.c_int(1)
        if prepared:
            extent = lib.test_provider_rhs_owner_bytes()
            owner, op = rhs.allocation(extent)
            assert lib.test_provider_rhs_prepare(views, op, extent, C.byref(epoch), 4) == 1
        before = lib.get_product_calls()
        for use in range(4 if prepared else 1):
            output = np.full((1, plan.heads, plan.query_rows, plan.depth), 0xDEAD, np.uint16)
            ov = fixture.view(output)
            if prepared:
                assert lib.prepared_run(views, C.byref(ov), pointer, size, op, extent, C.byref(epoch), use) == 1
            else:
                assert lib.run(views, C.byref(ov), pointer, size, 0) == 1
        callback_counts.append(lib.get_product_calls() - before)
        results.append(output)
        offset = pointer - raw.ctypes.data
        assert (raw[:offset] == 0xAC).all() and (raw[offset + size :] == 0xAC).all()
        if prepared:
            assert lib.prepared_run(views, C.byref(ov), pointer, size, op, extent, C.byref(epoch), 4) == 0
            offset = op - owner.ctypes.data
            assert (owner[:offset] == 0xAC).all() and (owner[offset + extent :] == 0xAC).all()
    assert np.array_equal(*results) and callback_counts[0] == 5 * callback_counts[1]
    assert callback_counts[1] == (4 if prepared else 1) * plan.heads * 8
    assert all(np.array_equal(a, b) for a, b in zip(xs, originals, strict=True))


@pytest.mark.parametrize("failure", [1, 2])
def test_failed_family_does_not_publish_partially_written_private_readout(pair, failure):
    plan, _, libraries = pair
    lib = libraries[1]
    xs = rhs.inputs(plan, 7)
    views = (fixture.View * 11)(*[fixture.view(x) for x in xs])
    size = lib.test_provider_workspace_bytes()
    raw, pointer = rhs.allocation(size)
    output = np.full((1, plan.heads, plan.query_rows, plan.depth), 0xDEAD, np.uint16)
    ov = fixture.view(output)
    assert lib.run(views, C.byref(ov), pointer, size, failure) == 0
    assert (output == 0xDEAD).all()
