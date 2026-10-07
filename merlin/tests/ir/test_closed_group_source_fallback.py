"""Retained source fallbacks keep complete arithmetic and a real upstream ABI."""

import hashlib
import json
import subprocess
from dataclasses import replace

import numpy as np
import pytest
from test_closed_group_writer import contract, prepared
from xdsl.dialects import builtin, func, memref

from merlin.llvmlower.closed_group_writer import (
    install_closed_group_writers,
    source_function_semantic_sha256,
    writer_contract_sha256,
)
from merlin.xdsl_dialects._common import text


def test_equal_complete_source_and_context_share_one_retained_fallback(tmp_path):
    ir, receipt, records, functions = prepared(tmp_path)
    contracts = {
        record["symbol"]: replace(contract(functions[record["symbol"]]), source_fallback_symbol="source_fallback")
        for record in records
    }
    result = install_closed_group_writers(ir, receipt, contracts)
    retained = next(
        op for op in ir.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "source_fallback"
    )
    assert retained.is_declaration is False
    assert retained.sym_visibility is None
    assert "llvm.emit_c_interface" in retained.attributes
    assert not any(name.startswith("merlin.source_group_") for name in retained.attributes)
    assert any(op.name == "math.fma" for op in retained.walk())
    (report,) = result["retained_source_fallbacks"]
    assert report["source_symbols"] == [record["symbol"] for record in records]
    assert report["source_semantic_sha256"] == report["retained_semantic_sha256"]
    assert report["c_interface_symbol"] == "_mlir_ciface_source_fallback"
    assert source_function_semantic_sha256(retained) == contracts[records[0]["symbol"]].source_semantic_sha256
    ir.verify()


@pytest.mark.parametrize(
    "name", ["external_group", "external_group_borrowed", "source_fallback", "_mlir_ciface_source_fallback"]
)
def test_fallback_collision_refuses_before_body_replacement(tmp_path, name):
    ir, receipt, records, functions = prepared(tmp_path)
    if name in ("source_fallback", "_mlir_ciface_source_fallback"):
        ir.body.block.add_op(
            memref.GlobalOp.get(builtin.StringAttr(name), builtin.MemRefType(builtin.f32, [1]), builtin.UnitAttr())
        )
    before = text(ir)
    supplied = replace(
        contract(functions[records[0]["symbol"]]),
        source_fallback_symbol=name if name.startswith("external") else "source_fallback",
    )
    with pytest.raises(ValueError, match="collides"):
        install_closed_group_writers(ir, receipt, {records[0]["symbol"]: supplied})
    assert text(ir) == before


def test_same_fallback_cannot_share_unequal_complete_source(tmp_path):
    ir, receipt, records, functions = prepared(tmp_path)
    # This attribute is numerically relevant and remains in the complete source
    # fingerprint; update the source receipt only to isolate dedup validation.
    other = functions[records[1]["symbol"]]
    other.attributes["source.numeric_contract"] = builtin.StringAttr("different")
    records[1]["function_sha256"] = hashlib.sha256(text(other).encode()).hexdigest()
    before = text(ir)
    contracts = {
        record["symbol"]: replace(
            contract(functions[record["symbol"]], "provider_" + str(index)), source_fallback_symbol="shared_fallback"
        )
        for index, record in enumerate(records)
    }
    with pytest.raises(ValueError, match="unequal semantics, ABI or context"):
        install_closed_group_writers(ir, receipt, contracts)
    assert text(ir) == before


def test_default_source_fallback_contract_hash_remains_compatible(tmp_path):
    from dataclasses import asdict

    ir, _, records, functions = prepared(tmp_path)
    value = contract(functions[records[0]["symbol"]])
    old = asdict(value)
    del old["private_workspaces"], old["source_fallback_symbol"]
    assert writer_contract_sha256(value) == hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()
    assert writer_contract_sha256(replace(value, source_fallback_symbol="fallback")) != writer_contract_sha256(value)


@pytest.mark.parametrize("optimization", ["-O0", "-O2"])
def test_actual_upstream_borrowed_writer_calls_original_source_fallback(tmp_path, optimization, upstream_host_tools):
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.pipeline import lower_to_llvm_ir
    from merlin.llvmlower.toolchain import clang

    ir, receipt, records, functions = prepared(tmp_path, emit_c_interface=True)
    original = text(ir, generic=True)
    selected = records[0]["symbol"]
    install_closed_group_writers(
        ir,
        receipt,
        {selected: replace(contract(functions[selected]), source_fallback_symbol="original_source_fallback")},
    )
    features = frozenset({"lower_fma_to_intrinsic"})
    llvm = lower_to_llvm_ir(
        text(ir, generic=True), workdir=tmp_path / "candidate_lower", vectorize=False, features=features
    )
    assert "define void @_mlir_ciface_original_source_fallback(ptr %0, ptr %1, ptr %2)" in llvm
    assert "@llvm.fma.f32" in llvm
    adapter = tmp_path / "fallback_adapter.c"
    adapter.write_text("""#include <stdint.h>
#include <string.h>
typedef struct{void*allocated,*aligned;int64_t offset,size[3],stride[3];}M;
void _mlir_ciface_original_source_fallback(M*,M*,M*);
void __real_free(void*);
static int calls,ownership_errors,active;
static void* protected[3];
int source_fallback_calls(void){return calls;}
int source_fallback_ownership_errors(void){return ownership_errors;}
void __wrap_free(void*p){
  if(active)for(int i=0;i<3;i++)if(p &&p==protected[i]){ownership_errors++;return;}
  __real_free(p);
}
void _mlir_ciface_external_group_borrowed(M*a,M*b,M*out){
  calls++; M before=*out;
  protected[0]=a->allocated;protected[1]=b->allocated;protected[2]=out->allocated;active=1;
  _mlir_ciface_original_source_fallback(a,b,out);
  if(memcmp(&before,out,sizeof(M)))ownership_errors++;
  active=0;
}
""")
    libraries = []
    for label, lowered in [
        ("candidate", llvm),
        ("control", lower_to_llvm_ir(original, workdir=tmp_path / "control_lower", vectorize=False, features=features)),
    ]:
        path = tmp_path / (label + ".ll")
        path.write_text(lowered)
        obj = tmp_path / (label + ".o")
        library = tmp_path / (label + optimization + ".so")
        subprocess.run([str(clang()), optimization, "-fPIC", "-c", str(path), "-o", str(obj)], check=True)
        sources = [str(obj), str(mlir_runtime_c())]
        if label == "candidate":
            sources.append(str(adapter))
        link = ["cc", optimization, "-fPIC", "-shared", *sources, "-lm"]
        if label == "candidate":
            link.append("-Wl,--wrap=free")
        subprocess.run([*link, "-o", str(library)], check=True)
        libraries.append(HostModel.load(str(library)))
    candidate, control = libraries
    generator = np.random.default_rng(413)
    for iteration in range(3):
        inputs = [generator.uniform(-4, 4, size=shape).astype(np.float32) for shape in [(2, 3, 4), (2, 4, 5)]]
        # Exactly represented BF16 raw words; compare the complete actual source
        # outputs, including cancellation and negative values, against itself.
        inputs = [(value.view(np.uint32) >> 16).astype(np.uint16) for value in inputs]
        saved = [value.copy() for value in inputs]
        outputs = []
        for invoke in (candidate, control):
            out = [np.full((2, 3, 5), 0x7FC1, np.uint16) for _ in range(2)]
            invoke([(value.ctypes.data, value.shape) for value in [*inputs, *out]])
            outputs.append(out)
        for actual, expected in zip(*outputs):
            np.testing.assert_array_equal(actual, expected)
        for actual, expected in zip(inputs, saved):
            np.testing.assert_array_equal(actual, expected)
        assert candidate.lib.source_fallback_calls() == iteration + 1
        assert candidate.lib.source_fallback_ownership_errors() == 0
