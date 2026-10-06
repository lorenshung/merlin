"""Read-only view acceptance retains the dense producer and exact numeric ABI."""

from dataclasses import replace

import pytest
from xdsl.dialects import tensor
from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr
from xdsl.parser import Parser

from merlin.llvmlower.fresh_tensor_writer import FreshTensorWriterContract, rewrite_fresh_tensor_writers
from merlin.llvmlower.segmented_input_acceptance import SegmentedInputContract, rewrite_segmented_inputs
from merlin.llvmlower.segmented_matrix_view import prove_segmented_matrix
from merlin.xdsl_dialects._common import make_context, text

SOURCE = """module {
func.func @forward(%a:tensor<1x3x5x2xi8>,%b:tensor<2x3xi8>)
    -> (tensor<1x3x5x2xi8>,tensor<4x3xi8>) attributes {llvm.emit_c_interface} {
 %buffer=tensor.empty():tensor<1x3x5x2xi8>
 %owner=func.call @produce(%a,%buffer)
     :(tensor<1x3x5x2xi8>,tensor<1x3x5x2xi8>)->tensor<1x3x5x2xi8>
 %slice="tensor.extract_slice"(%owner) <{
     static_offsets=array<i64: 0,0,1,0>, static_sizes=array<i64: 1,2,2,2>,
     static_strides=array<i64: 1,2,2,1>, operandSegmentSizes=array<i32: 1,0,0,0>
 }> : (tensor<1x3x5x2xi8>)->tensor<1x2x2x2xi8>
 %flat="tensor.collapse_shape"(%slice) <{reassociation=[[0 : i64,1 : i64,2 : i64,3 : i64]]}>
     : (tensor<1x2x2x2xi8>)->tensor<8xi8>
 %view="tensor.expand_shape"(%flat) <{reassociation=[[0 : i64,1 : i64]],
     static_output_shape=array<i64: 4,2>}> : (tensor<8xi8>)->tensor<4x2xi8>
 %output=tensor.empty():tensor<4x3xi8>
 %result=func.call @consume(%view,%b,%output)
     :(tensor<4x2xi8>,tensor<2x3xi8>,tensor<4x3xi8>)->tensor<4x3xi8>
 return %owner,%result:tensor<1x3x5x2xi8>,tensor<4x3xi8>
}
func.func private @produce(tensor<1x3x5x2xi8>,tensor<1x3x5x2xi8>)
    ->tensor<1x3x5x2xi8> attributes {llvm.emit_c_interface}
func.func private @consume(tensor<4x2xi8>,tensor<2x3xi8>,tensor<4x3xi8>)
    ->tensor<4x3xi8> attributes {llvm.emit_c_interface, merlin.numeric_contract="exact"}
}"""


def fixture():
    module = Parser(make_context(tensor.Tensor), SOURCE).parse_module()
    for op in module.body.block.ops:
        if op.name != "func.func" or op.body.blocks:
            continue
        accesses = ("read", "write") if op.sym_name.data == "produce" else ("read", "read", "write")
        op.properties["arg_attrs"] = ArrayAttr(
            [DictionaryAttr({"bufferization.access": StringAttr(access)}) for access in accesses]
        )
    module.verify()
    view = next(op.results[0] for op in module.walk() if op.name == "tensor.expand_shape")
    proof = prove_segmented_matrix(view)
    contract = SegmentedInputContract("consume", 0, "consume_segmented", proof.address)
    writers = [
        FreshTensorWriterContract("produce", 1, (1,), "produce_borrowed", 64),
        FreshTensorWriterContract("consume", 2, (2,), "consume_borrowed", 64),
    ]
    return module, proof, contract, writers


def test_empty_selection_is_byte_preserving():
    module, _, _, _ = fixture()
    before = text(module, generic=True)
    assert rewrite_segmented_inputs(module, [], fresh_writers=[]) == []
    assert text(module, generic=True) == before


def test_acceptance_preserves_owner_fanout_attributes_and_other_arguments():
    module, proof, contract, writers = fixture()
    call = next(op for op in module.walk() if op.name == "func.call" and op.callee.root_reference.data == "consume")
    other = tuple(call.operands[1:])
    report = rewrite_segmented_inputs(module, [contract], fresh_writers=writers)
    assert call.arguments[0] is proof.owner
    assert tuple(call.arguments[1:]) == other
    assert report[0]["owner_producers"] == ["produce"]
    declarations = {op.sym_name.data: op for op in module.body.block.ops}
    old, accepted = declarations["consume"], declarations["consume_segmented"]
    assert accepted.attributes["merlin.numeric_contract"] == old.attributes["merlin.numeric_contract"]
    assert accepted.arg_attrs == old.arg_attrs
    assert accepted.function_type.outputs == old.function_type.outputs
    assert sum(1 for _ in proof.owner.uses) == 3  # source slice, accepted consumer and live returned owner
    updated = [writers[0], replace(writers[1], symbol=contract.accepted_symbol, borrowed_symbol="segmented_borrowed")]
    rewrite_fresh_tensor_writers(module, updated)
    assert "read_only" in text(module, generic=True)
    module.verify()


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"accepted_symbol": "produce"}, "collision"),
        ({"argument": 2}, "read-only"),
        ({"argument": -1}, "outside"),
        ({"address": None}, "typed"),
    ],
)
def test_invalid_acceptance_does_not_edit_any_source(change, reason):
    module, _, contract, writers = fixture()
    before = text(module, generic=True)
    with pytest.raises(ValueError, match=reason):
        rewrite_segmented_inputs(module, [replace(contract, **change)], fresh_writers=writers)
    assert text(module, generic=True) == before


def test_map_disagreement_and_missing_producer_refuse_atomically():
    module, _, contract, writers = fixture()
    before = text(module, generic=True)
    bad = replace(contract, address=replace(contract.address, origin=0))
    with pytest.raises(ValueError, match="differs"):
        rewrite_segmented_inputs(module, [bad], fresh_writers=writers)
    with pytest.raises(ValueError, match="producer"):
        rewrite_segmented_inputs(module, [contract], fresh_writers=writers[1:])
    assert text(module, generic=True) == before


def test_later_invalid_selection_does_not_partially_retarget():
    module, _, contract, writers = fixture()
    before = text(module, generic=True)
    with pytest.raises(ValueError, match="duplicate"):
        rewrite_segmented_inputs(module, [contract, contract], fresh_writers=writers)
    assert text(module, generic=True) == before


def test_native_borrowed_input_retains_lifetime_and_input_bytes(tmp_path):
    import subprocess

    import numpy as np

    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.pipeline import lower_to_llvm_ir
    from merlin.llvmlower.toolchain import clang

    module, _, contract, writers = fixture()
    rewrite_segmented_inputs(module, [contract], fresh_writers=writers)
    rewrite_fresh_tensor_writers(
        module,
        [
            writers[0],
            replace(writers[1], symbol=contract.accepted_symbol, borrowed_symbol="segmented_borrowed"),
        ],
    )
    llvm = lower_to_llvm_ir(text(module, generic=True), workdir=tmp_path / "lower", vectorize=False)
    assert "call void @free(" in llvm
    source, adapter = tmp_path / "model.ll", tmp_path / "adapter.c"
    source.write_text(llvm)
    adapter.write_text("""#include <stdint.h>
#include <stdlib.h>
typedef struct {void *allocated,*aligned;int64_t offset,size[2],stride[2];} M2;
typedef struct {void *allocated,*aligned;int64_t offset,size[4],stride[4];} M4;
void _mlir_ciface_produce_borrowed(M4*a,M4*out) {
 if(a->aligned==out->aligned) abort();
 for(int i=0;i<30;i++)((int8_t*)out->aligned)[out->offset+i]=((int8_t*)a->aligned)[a->offset+i];
}
void _mlir_ciface_segmented_borrowed(M4*a,M2*b,M2*out) {
 if(a->size[0]!=1 || a->size[1]!=3 || a->size[2]!=5 || a->size[3]!=2)abort();
 if(a->stride[0]!=30 || a->stride[1]!=10 || a->stride[2]!=2 || a->stride[3]!=1)abort();
 for(int m=0;m<4;m++)for(int n=0;n<3;n++) {
  int acc=0;for(int k=0;k<2;k++)
   acc+=((int8_t*)a->aligned)[a->offset+(m/2)*20+(m%2)*4+2+k]*((int8_t*)b->aligned)[b->offset+k*3+n];
  ((int8_t*)out->aligned)[out->offset+m*3+n]=(int8_t)acc;
 }
}
""")
    obj, library = tmp_path / "model.o", tmp_path / "model.so"
    subprocess.run([str(clang()), "-O2", "-fPIC", "-c", str(source), "-o", str(obj)], check=True)
    subprocess.run(
        ["cc", "-O2", "-fPIC", "-shared", str(obj), str(adapter), str(mlir_runtime_c()), "-o", str(library)], check=True
    )
    a = np.arange(30, dtype=np.int8).reshape(1, 3, 5, 2)
    b = np.array([[-1, 2, 1], [3, 0, -2]], dtype=np.int8)
    owner, output = np.zeros_like(a), np.zeros((4, 3), dtype=np.int8)
    invoke = HostModel.load(str(library))
    expected = a[:, ::2, 1::2, :].reshape(4, 2).astype(np.int32) @ b.astype(np.int32)
    for _ in range(5):
        invoke([(value.ctypes.data, value.shape) for value in (a, b, owner, output)])
        np.testing.assert_array_equal(owner, a)
        np.testing.assert_array_equal(output, expected.astype(np.int8))
        np.testing.assert_array_equal(a, np.arange(30, dtype=np.int8).reshape(a.shape))
