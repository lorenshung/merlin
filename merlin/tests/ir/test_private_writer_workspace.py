"""Private byte scratch keeps ownership, output identity and typed ABI explicit."""

import hashlib
import json
import subprocess
from dataclasses import replace

import numpy as np
import pytest
from test_fresh_tensor_writer import contract, fixture
from xdsl.dialects import func, memref

from merlin.llvmlower.fresh_tensor_writer import PrivateWorkspaceContract, rewrite_fresh_tensor_writers
from merlin.xdsl_dialects._common import text


def workspace(size=37, alignment=64):
    return PrivateWorkspaceContract(size, alignment, "a" * 64, True, True, True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("bytes", 0),
        ("bytes", -1),
        ("bytes", True),
        ("bytes", 1 << 63),
        ("alignment", 3),
        ("alignment", True),
        ("bytes", (1 << 63) - 1),
        ("effect_witness_sha256", "missing"),
        ("initializes_before_read", False),
        ("borrowed_noescape", False),
        ("synchronous_before_return", False),
    ],
)
def test_invalid_private_workspace_refuses_transactionally(field, value):
    module = fixture()
    before = text(module)
    bad = replace(workspace(), **{field: value})
    with pytest.raises(ValueError, match="workspace"):
        rewrite_fresh_tensor_writers(module, [replace(contract(), private_workspaces=(bad,))])
    assert text(module) == before


def test_private_workspace_is_not_a_tensor_argument_or_result():
    module = fixture()
    report = rewrite_fresh_tensor_writers(module, [replace(contract(), private_workspaces=(workspace(),))])
    wrapper = next(op for op in module.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "writer")
    borrowed = next(
        op for op in module.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "borrowed"
    )
    assert len(wrapper.function_type.inputs) == 3 and len(wrapper.function_type.outputs) == 1
    assert len(borrowed.function_type.inputs) == 4 and not borrowed.function_type.outputs
    operations = list(wrapper.body.block.ops)
    call = next(op for op in operations if isinstance(op, func.CallOp))
    private = call.arguments[-1]
    assert isinstance(private.owner, memref.AllocOp) and private.type.get_shape() == (37,)
    releases = [op for op in operations if isinstance(op, memref.DeallocOp)]
    assert releases == []  # Normal upstream owns release after bufferization.
    assert report[0]["allocator_reclamation"] == "runtime_policy_not_inferred"
    assert report[0]["private_workspaces"][0]["effect_witness_sha256"] == "a" * 64


def test_workspace_contract_hash_keeps_empty_default_compatible(tmp_path):
    from dataclasses import asdict

    from test_consumer_observed_group_writer import prepare

    from merlin.llvmlower.closed_group_writer import writer_contract_sha256

    _, _, contracts, _ = prepare(tmp_path)
    value = next(iter(contracts.values()))
    old = asdict(value)
    del old["private_workspaces"]
    del old["source_fallback_symbol"]
    assert writer_contract_sha256(value) == hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()
    assert writer_contract_sha256(replace(value, private_workspaces=(workspace(),))) != writer_contract_sha256(value)


@pytest.mark.parametrize("optimization", ["-O0", "-O2"])
def test_actual_upstream_private_workspace_native(tmp_path, optimization, upstream_host_tools):
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.pipeline import lower_to_llvm_ir
    from merlin.llvmlower.toolchain import clang

    module = fixture()
    rewrite_fresh_tensor_writers(module, [replace(contract(), private_workspaces=(workspace(),))])
    llvm = lower_to_llvm_ir(text(module, generic=True), workdir=tmp_path / "lower", vectorize=False)
    assert "call void @free(" in llvm
    source = tmp_path / "model.ll"
    source.write_text(llvm)
    adapter = tmp_path / "adapter.c"
    adapter.write_text("""#include <stdint.h>
#include <stdlib.h>
typedef struct{void*allocated,*aligned;int64_t offset,size[1],stride[1];}M;
static void*live_private;static int releases,inside;
void __real_free(void*);
void __wrap_free(void*p){if(live_private &&p==live_private){if(inside)abort();releases++;live_private=0;}__real_free(p);}
int private_releases(void){return releases;}
void _mlir_ciface_borrowed(M*a,M*s,M*out,M*private){
 inside=1;live_private=private->allocated;
 if(private->size[0]!=37 || private->stride[0]!=1 || private->offset!=0 || (uintptr_t)private->aligned%64)abort();
 if(private->aligned==a->aligned || private->aligned==s->aligned || private->aligned==out->aligned)abort();
 for(int i=0;i<37;i++)((uint8_t*)private->aligned)[i]=(uint8_t)(i+5);
 for(int i=0;i<7;i++)((int32_t*)s->aligned)[i]=i+3;
 for(int i=0;i<4;i++)((int8_t*)out->aligned)[i]=(int8_t)(((int8_t*)a->aligned)[a->offset+i*a->stride[0]]+((uint8_t*)private->aligned)[i]);
 inside=0;
}
""")
    obj = tmp_path / "model.o"
    library = tmp_path / ("private_workspace_" + optimization + ".so")
    subprocess.run([str(clang()), optimization, "-fPIC", "-c", str(source), "-o", str(obj)], check=True)
    subprocess.run(
        [
            "cc",
            optimization,
            "-fPIC",
            "-shared",
            str(obj),
            str(adapter),
            str(mlir_runtime_c()),
            "-Wl,--wrap=free",
            "-o",
            str(library),
        ],
        check=True,
    )
    invoke = HostModel.load(str(library))
    original = np.array([-128, -1, 0, 110], np.int8)
    for iteration in range(20):
        first = np.full(4, 77, np.int8)
        second = np.full(4, 77, np.int8)
        invoke([(v.ctypes.data, v.shape) for v in [original, first, second]])
        np.testing.assert_array_equal(original, [-128, -1, 0, 110])
        np.testing.assert_array_equal(first, original)
        np.testing.assert_array_equal(second, [-123, 5, 7, 118])
        assert invoke.lib.private_releases() == iteration + 1
