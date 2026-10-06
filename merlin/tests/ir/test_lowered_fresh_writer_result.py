"""Actual result out-parameter owners, independent of earlier writer buffers."""

from dataclasses import replace

import pytest
from xdsl.context import Context
from xdsl.dialects import llvm
from xdsl.dialects.builtin import Builtin
from xdsl.parser import Parser

from merlin.llvmlower.fresh_tensor_writer import (
    FreshTensorWriterContract,
    prove_lowered_writer_result,
)


def fixture(*, changed=None, old_destination=False):
    types = ["!llvm.ptr", "!llvm.ptr", "i64", "i64", "i64", "i64", "i64"]
    arguments = [f"%x{i}" for i in range(28)]
    wrapper_args = ",".join(f"{a}:{types[i % 7]}" for i, a in enumerate(arguments))
    forwarded = arguments[:14] + arguments[14 : 21 if old_destination else 14]
    if not old_destination:
        forwarded += arguments[21:]
    args = ",".join(forwarded)
    descriptor = ["%a", "%a", "%zero", "%m", "%n", "%n", "%one"]
    call_args = descriptor + [v.replace("%a", "%b") for v in descriptor]
    call_args += [v.replace("%a", "%unused") for v in descriptor]
    call_args += [v.replace("%a", "%out") for v in descriptor]
    if changed:
        for index, value in changed.items():
            call_args[index] = value
    call = ",".join(call_args)
    text = f"""builtin.module {{
llvm.func @malloc(i64) -> !llvm.ptr
llvm.func @borrowed({",".join(types * 3)})
llvm.func @writer({wrapper_args}) {{
 llvm.call @borrowed({args}):({",".join(types * 3)})->()
 llvm.return
}}
llvm.func @caller() {{
 %zero=llvm.mlir.constant(0:i64):i64
 %one=llvm.mlir.constant(1:i64):i64
 %m=llvm.mlir.constant(32:i64):i64
 %n=llvm.mlir.constant(64:i64):i64
 %bytes=llvm.mlir.constant(2048:i64):i64
 %small=llvm.mlir.constant(16:i64):i64
 %a=llvm.call @malloc(%bytes):(i64)->!llvm.ptr
 %b=llvm.call @malloc(%bytes):(i64)->!llvm.ptr
 %unused=llvm.call @malloc(%bytes):(i64)->!llvm.ptr
 %out=llvm.call @malloc(%bytes):(i64)->!llvm.ptr
 %short=llvm.call @malloc(%small):(i64)->!llvm.ptr
 llvm.call @writer({call}):({",".join(types * 4)})->()
 llvm.return
}}
}}"""
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(llvm.LLVM)
    module = Parser(ctx, text).parse_module()
    module.verify()
    functions = {op.sym_name.data: op for op in module.body.block.ops}
    callsite = next(
        op
        for op in functions["caller"].walk()
        if isinstance(op, llvm.CallOp) and op.callee.root_reference.data == "writer"
    )
    return module, functions["writer"], callsite


CONTRACT = FreshTensorWriterContract("writer", 2, (2,), "borrowed", 64)


def prove(wrapper, callsite, contract=CONTRACT, alignment=64):
    return prove_lowered_writer_result(
        wrapper,
        callsite,
        contract,
        argument_shapes=((32, 64),) * 3,
        element_bytes=(1, 1, 1),
        allocator_alignment=alignment,
    )


def test_final_result_owner_is_used_and_no_ir_changes():
    module, wrapper, callsite = fixture()
    before = str(module)
    witness = prove(wrapper, callsite)
    assert witness["actual_output_parameter_start"] == 21
    assert witness["distinct_direct_output_owner"]
    assert witness["allocator_runtime_closure"] == "REQUIRED_SEPARATELY"
    assert witness["allocator_alignment_assumption"] == 64
    assert witness["descriptors"][2]["bytes"] == 2048
    assert str(module) == before


@pytest.mark.parametrize(
    "changes",
    [
        {21: "%a", 22: "%a"},
        {22: "%unused"},
        {23: "%one"},
        {24: "%n"},
        {26: "%one"},
        {21: "%short", 22: "%short"},
    ],
)
def test_alias_subview_stride_bounds_and_capacity_refuse(changes):
    _, wrapper, callsite = fixture(changed=changes)
    with pytest.raises(ValueError, match="owner/ABI"):
        prove(wrapper, callsite)


def test_earlier_writer_allocation_does_not_prove_final_destination():
    _, wrapper, callsite = fixture(old_destination=True)
    with pytest.raises(ValueError, match="owner/ABI"):
        prove(wrapper, callsite)


@pytest.mark.parametrize("alignment", [None, 0, 3, True, 1 << 63])
def test_unknown_runtime_alignment_refuses(alignment):
    _, wrapper, callsite = fixture()
    with pytest.raises(ValueError):
        prove(wrapper, callsite, alignment=alignment)


def test_wrong_borrowed_symbol_or_full_writer_refuses():
    _, wrapper, callsite = fixture()
    for contract in (replace(CONTRACT, borrowed_symbol="other"), replace(CONTRACT, fully_written_arguments=(1, 2))):
        with pytest.raises(ValueError):
            prove(wrapper, callsite, contract)
