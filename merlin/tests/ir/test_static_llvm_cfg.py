import pytest
from xdsl.context import Context
from xdsl.dialects import llvm
from xdsl.dialects.builtin import Builtin, IntegerAttr, i32
from xdsl.parser import Parser

from merlin.llvmlower.static_llvm_cfg import StaticInt, StaticPointer, StaticTraceError, trace_static_function


def parse(body, args=""):
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(llvm.LLVM)
    module = Parser(
        ctx, "builtin.module { llvm.func @observe(i8) llvm.func @trace(" + args + ") {\n" + body + "\n} }"
    ).parse_module()
    module.verify()
    return list(module.body.block.ops)[1]


def trace(fn, args=(), limit=100):
    return list(
        trace_static_function(
            fn, args, observe=lambda op: isinstance(op, llvm.CallOp), pointer_index_bits=64, max_steps=limit
        )
    )


def test_declared_width_wrap_controls_signed_branch_and_block_arguments():
    fn = parse("""
 %start = llvm.mlir.constant(126 : i8) : i8
 %one = llvm.mlir.constant(1 : i8) : i8
 %stop = llvm.mlir.constant(-126 : i8) : i8
 llvm.br ^loop(%start : i8)
 ^loop(%iv: i8):
 llvm.call @observe(%iv) : (i8) -> ()
 %next = llvm.add %iv, %one : i8
 %more = llvm.icmp "ne" %next, %stop : i8
 llvm.cond_br %more, ^loop(%next : i8), ^exit
 ^exit:
 llvm.return
""")
    assert [s.inputs[0].signed for s in trace(fn)] == [126, 127, -128, -127]


@pytest.mark.parametrize("predicate,expected", [("slt", True), ("ult", False), ("eq", False), ("ne", True)])
def test_signed_and_unsigned_predicates(predicate, expected):
    fn = parse(
        """
 %neg = llvm.mlir.constant(-1 : i8) : i8
 %zero = llvm.mlir.constant(0 : i8) : i8
 %cmp = llvm.icmp "PRED" %neg, %zero : i8
 llvm.cond_br %cmp, ^yes, ^no
 ^yes:
 llvm.call @observe(%neg) : (i8) -> ()
 llvm.return
 ^no:
 llvm.return
""".replace("PRED", predicate)
    )
    assert bool(trace(fn)) == expected


def test_pointer_indices_use_signed_offset_and_explicit_element_size():
    # Observe pointer with a result-free store. No memory is read or mutated by the tracer.
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(llvm.LLVM)
    module = Parser(
        ctx,
        """builtin.module { llvm.func @trace(%arg: !llvm.ptr) {
      %minus = llvm.mlir.constant(-1 : i64) : i64
      %value = llvm.mlir.constant(7 : i32) : i32
      %ptr = llvm.getelementptr %arg[%minus] : (!llvm.ptr,i64) -> !llvm.ptr,i32
      llvm.store %value, %ptr : i32,!llvm.ptr
      llvm.return
    } }""",
    ).parse_module()
    module.verify()
    steps = list(
        trace_static_function(
            module.body.block.first_op,
            [StaticPointer("allocation", StaticInt(8, 64))],
            observe=lambda op: isinstance(op, llvm.StoreOp),
            pointer_index_bits=64,
        )
    )
    assert steps[0].inputs[1] == StaticPointer("allocation", StaticInt(4, 64))


def test_dynamic_load_and_unbounded_loop_refuse_complete_trace():
    fn = parse(
        """%x = llvm.load %arg : !llvm.ptr -> i8
 llvm.call @observe(%x) : (i8) -> ()
 llvm.return""",
        "%arg: !llvm.ptr",
    )
    with pytest.raises(StaticTraceError, match="llvm.load"):
        trace(fn, [StaticPointer("allocation", StaticInt(0, 64))])
    loop = parse("llvm.br ^loop\n^loop:\nllvm.br ^loop")
    with pytest.raises(StaticTraceError, match="step limit"):
        trace(loop, limit=10)


def test_opaque_result_is_not_assumed_constant():
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(llvm.LLVM)
    module = Parser(
        ctx,
        """builtin.module { llvm.func @unknown() -> i8
    llvm.func @trace() { %x = llvm.call @unknown() : () -> i8 llvm.return } }""",
    ).parse_module()
    with pytest.raises(StaticTraceError, match="result-free"):
        trace(list(module.body.block.ops)[1])


def test_inbounds_without_storage_extent_proof_is_refused():
    fn = parse(
        """%zero = llvm.mlir.constant(0 : i64) : i64
    %ptr = llvm.getelementptr inbounds %arg[%zero] : (!llvm.ptr,i64) -> !llvm.ptr,i8
    llvm.return""",
        "%arg: !llvm.ptr",
    )
    with pytest.raises(StaticTraceError, match="extent proof"):
        trace(fn, [StaticPointer("allocation", StaticInt(0, 64))])


def test_poison_flags_and_mismatched_argument_width_are_refused():
    fn = parse(
        """%one = llvm.mlir.constant(1 : i8) : i8
    %sum = llvm.add %arg, %one : i8 llvm.return""",
        "%arg: i8",
    )
    with pytest.raises(StaticTraceError, match="argument type differs"):
        trace(fn, [StaticInt(1, 64)])
    addition = next(op for op in fn.walk() if isinstance(op, llvm.AddOp))
    addition.properties["overflowFlags"] = IntegerAttr(1, i32)
    with pytest.raises(StaticTraceError, match="poison-producing"):
        trace(fn, [StaticInt(1, 8)])
