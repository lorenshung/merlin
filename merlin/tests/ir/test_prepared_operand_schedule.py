"""Prepared data reuse is an explicit source schedule, never a pointer cache."""

import pytest
from xdsl.dialects import bufferization, builtin, func, memref, tensor
from xdsl.ir import Block, Region

from merlin.llvmlower.prepared_operand_owner import PreparedRepresentation, PreparedStorageSpan
from merlin.llvmlower.prepared_operand_schedule import install_prepared_borrows, plan_prepared_borrows
from merlin.llvmlower.tensor_preparation_identity import TensorPreparationRequest, find_tensor_preparation_opportunities


def fixture(strided=False):
    typ = builtin.TensorType(builtin.f32, [3, 17])
    rb = Block(arg_types=[typ])
    rb.add_op(func.ReturnOp(rb.args[0]))
    reader = func.FuncOp("source_read", ([typ], [typ]), Region(rb), visibility="private")
    input_type = builtin.TensorType(builtin.f32, [3, 19]) if strided else typ
    block = Block(arg_types=[input_type])
    calls = []
    for _ in range(4):
        value = block.args[0]
        if strided:
            view = tensor.ExtractSliceOp.from_static_parameters(value, [0, 0], [3, 17])
            block.add_op(view)
            value = view.result
        call = func.CallOp("source_read", [value], [typ])
        block.add_op(call)
        calls.append(call)
    block.add_op(func.ReturnOp(calls[-1].results[0]))
    owner = func.FuncOp("forward", ([input_type], [typ]), Region(block))
    owner.attributes["llvm.emit_c_interface"] = builtin.UnitAttr()
    module = builtin.ModuleOp([reader, owner])
    opportunities = find_tensor_preparation_opportunities(
        [TensorPreparationRequest(call.arguments[0], call, "1" * 64) for call in calls]
    )
    rep = PreparedRepresentation("1" * 64, "2" * 64, "3" * 64, (PreparedStorageSpan("representation", 4096, 64),))
    plan = plan_prepared_borrows(module, [opportunities], rep)
    # Minimal post-writer-installation call tree. The physical validator below
    # is test-only; this fixture makes no external implementation proof claim.
    wb = Block(arg_types=[typ])
    buf = bufferization.ToBufferOp.build(
        operands=[wb.args[0]],
        result_types=[builtin.MemRefType(builtin.f32, [3, 17])],
        properties={"read_only": builtin.UnitAttr()},
    )
    wb.add_ops([buf, func.CallOp("raw_borrow", [buf.results[0]], []), func.ReturnOp(wb.args[0])])
    wrapper = func.FuncOp("writer", ([typ], [typ]), Region(wb), visibility="private")
    raw = func.FuncOp("raw_borrow", ([buf.results[0].type], []), Region(), visibility="private")
    raw.attributes["llvm.emit_c_interface"] = builtin.UnitAttr()
    new = Block(arg_types=[typ])
    call = func.CallOp("writer", [new.args[0]], [typ])
    new.add_ops([call, func.ReturnOp(call.results[0])])
    reader.detach_region(0)
    reader.add_region(Region(new))
    module.body.block.add_ops([raw, wrapper])
    module.verify()
    return module, plan, calls, [dict(symbol="writer", borrowed_symbol="raw_borrow")]


def install(module, plan, reports, validator=lambda *args: None, materialize=False):
    return install_prepared_borrows(
        module,
        plan,
        reports,
        prepare_symbol="prepare",
        borrowed_symbol="prepared_borrow",
        validate_physical=validator,
        materialize_shared_views=materialize,
    )


def test_one_pool_explicit_prepare_and_four_constant_leases():
    module, plan, calls, reports = fixture()
    before = plan.owner.function_type
    result = install(module, plan, reports)
    assert result["allocations"] == 1 and result["owner_bytes"] == 4097
    assert result["epochs"] == 1 and result["consumers"] == 4
    assert plan.owner.function_type == before
    allocations = [op for op in plan.owner.body.block.ops if isinstance(op, memref.AllocOp)]
    prepares = [
        op
        for op in plan.owner.body.block.ops
        if isinstance(op, func.CallOp) and op.callee.root_reference.data == "prepare"
    ]
    assert len(allocations) == len(prepares) == 1
    ops = list(plan.owner.body.block.ops)
    assert ops.index(prepares[0]) < ops.index(calls[0])
    assert [call.arguments[-1].owner.value.value.data for call in calls] == [0, 1, 2, 3]
    assert len({call.arguments[-3] for call in calls}) == 1
    module.verify()


@pytest.mark.parametrize("bad", ["physical", "input", "order", "unbound"])
def test_unknown_or_stale_binding_refuses_without_mutation(bad):
    module, plan, calls, reports = fixture()
    validator = lambda *args: None
    if bad == "physical":
        validator = lambda *args: True
    elif bad == "input":
        calls[1].operands = [calls[0].results[0]]
    elif bad == "order":
        block = plan.owner.body.block
        block.detach_op(calls[2])
        block.insert_op_before(calls[2], calls[0])
    else:
        extra = func.CallOp("source_read", [plan.owner.body.block.args[0]], [calls[0].results[0].type])
        plan.owner.body.block.insert_op_before(extra, calls[0])
    with pytest.raises(ValueError):
        install(module, plan, reports, validator)
    assert not any(isinstance(op, memref.AllocOp) for op in plan.owner.body.block.ops)


@pytest.mark.parametrize("optimization", ["-O0", "-O2"])
@pytest.mark.parametrize("strided", [False, True])
def test_actual_upstream_prepare_borrow_owner_and_release(tmp_path, optimization, strided):
    import subprocess

    import numpy as np

    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.pipeline import lower_to_llvm_ir
    from merlin.llvmlower.toolchain import clang
    from merlin.xdsl_dialects._common import text

    module, plan, calls, reports = fixture(strided)
    install(module, plan, reports, materialize=strided)
    llvm = lower_to_llvm_ir(text(module, generic=True), workdir=tmp_path / "lower", vectorize=False)
    ll = tmp_path / "model.ll"
    ll.write_text(llvm)
    adapter = tmp_path / "adapter.c"
    adapter.write_text(r"""
#include <stdint.h>
#include <stdlib.h>
typedef struct{void*allocated,*aligned;int64_t offset,size[2],stride[2];}R;
typedef struct{void*allocated,*aligned;int64_t offset,size[1],stride[1];}W;
static void*owned;static void*prepared_input;static int prepares,borrows,releases;
void __real_free(void*);
void __wrap_free(void*p){if(p==owned){if(borrows!=prepares*4)abort();owned=0;releases++;}__real_free(p);}
int counts(int n){return n==0?prepares:n==1?borrows:releases;}
void _mlir_ciface_prepare_borrowed(R*a,W*w,int64_t epoch,int64_t uses){
 if(owned||epoch!=0||uses!=4||w->size[0]!=4097||w->offset||w->stride[0]!=1||(uintptr_t)w->aligned%64||w->aligned==a->aligned)abort();
 owned=w->allocated;prepared_input=a->aligned;*(int64_t*)w->aligned=0;
 float*prepared=(float*)((char*)w->aligned+64);
 for(int r=0;r<3;r++)for(int c=0;c<17;c++)prepared[r*17+c]=2*((float*)a->aligned)[a->offset+r*a->stride[0]+c*a->stride[1]];
 prepares++;
}
void _mlir_ciface_prepared_borrow(R*a,W*w,int64_t epoch,int64_t ordinal){
 if(prepared_input!=a->aligned||owned!=w->allocated||epoch!=0||*(int64_t*)w->aligned!=ordinal||ordinal<0||ordinal>=4)abort();
 const float*prepared=(const float*)((const char*)w->aligned+64);
 for(int r=0;r<3;r++)for(int c=0;c<17;c++)if(prepared[r*17+c]!=2*((float*)a->aligned)[a->offset+r*a->stride[0]+c*a->stride[1]])abort();
 (*(int64_t*)w->aligned)++;borrows++;
}
""")
    obj = tmp_path / "model.o"
    lib = tmp_path / ("prepared_" + optimization + "_" + str(strided) + ".so")
    subprocess.run([str(clang()), optimization, "-fPIC", "-c", str(ll), "-o", str(obj)], check=True)
    subprocess.run(
        [
            "cc",
            optimization,
            "-shared",
            "-fPIC",
            str(obj),
            str(adapter),
            str(mlir_runtime_c()),
            "-Wl,--wrap=free",
            "-o",
            str(lib),
        ],
        check=True,
    )
    model = HostModel.load(str(lib))
    original = np.arange(57 if strided else 51, dtype=np.float32).reshape(3, 19 if strided else 17)
    for i in range(3):
        output = np.full((3, 17), -900, dtype=np.float32)
        model([(x.ctypes.data, x.shape) for x in (original, output)])
        np.testing.assert_array_equal(output, original[:, :17])
        np.testing.assert_array_equal(
            original, np.arange(57 if strided else 51, dtype=np.float32).reshape(original.shape)
        )
        assert [model.lib.counts(n) for n in range(3)] == [i + 1, 4 * (i + 1), i + 1]


def test_materialization_requires_explicit_boolean_and_exact_read_positions():
    module, plan, calls, reports = fixture(True)
    with pytest.raises(ValueError, match="explicit boolean"):
        install(module, plan, reports, materialize=1)
    result = install(module, plan, reports, materialize=True)
    assert result["shared_materializations"] == 1
    assert len({call.arguments[0] for call in calls}) == 1
    module.verify()


@pytest.mark.parametrize("mutation", ["strict_context", "global_symbol", "forged_extent"])
def test_preflight_refuses_context_symbol_or_allocation_changes_transactionally(mutation):
    from dataclasses import replace

    from merlin.xdsl_dialects._common import text

    module, plan, calls, reports = fixture()
    if mutation == "strict_context":
        plan.owner.attributes["strictfp"] = builtin.UnitAttr()
    elif mutation == "global_symbol":
        # A non-function symbol must reserve the generated declaration name too.
        module.body.block.add_op(
            memref.GlobalOp.get(
                builtin.StringAttr("prepare"),
                builtin.MemRefType(builtin.i8, [1]),
                builtin.UnitAttr(),
                sym_visibility=builtin.StringAttr("private"),
            )
        )
    else:
        plan = replace(plan, capacity=plan.capacity + 1)
    before = text(module, generic=True)
    with pytest.raises(ValueError):
        install(module, plan, reports)
    assert text(module, generic=True) == before


def test_aligned_pool_overflow_refuses_before_installation():
    from test_prepared_operand_effects import fixture as source_fixture

    module, _, _, calls, _ = source_fixture()
    # Each payload is representable, but the aligned allocation request is not.
    requests = [TensorPreparationRequest(c.arguments[0], c, "1" * 64) for c in calls]
    opportunities = find_tensor_preparation_opportunities(requests)
    representation = PreparedRepresentation(
        "1" * 64, "2" * 64, "3" * 64, (PreparedStorageSpan("large", (1 << 63) - 64, 128),)
    )
    with pytest.raises(ValueError, match="overflows"):
        plan_prepared_borrows(module, [opportunities], representation)


def test_new_unknown_effect_between_borrows_refuses_before_mutation():
    from merlin.xdsl_dialects._common import text

    module, plan, calls, reports = fixture()
    module.body.block.add_op(func.FuncOp("unknown_effect", ([], []), Region(), visibility="private"))
    plan.owner.body.block.insert_op_before(func.CallOp("unknown_effect", [], []), calls[2])
    module.verify()
    before = text(module)
    with pytest.raises(ValueError, match="unproved owner operation"):
        install(module, plan, reports)
    assert text(module) == before


def pooled_fixture():
    from dataclasses import asdict

    from test_private_writer_workspace import workspace

    from merlin.llvmlower.private_workspace_pool import pool_private_writer_workspaces

    module, plan, calls, reports = fixture()
    wrapper = next(op for op in module.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "writer")
    borrowed = next(
        op for op in module.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "raw_borrow"
    )
    call = next(op for op in wrapper.body.block.ops if isinstance(op, func.CallOp))
    contract = workspace()
    allocation = memref.AllocOp.get(builtin.i8, shape=[contract.bytes], alignment=contract.alignment)
    wrapper.body.block.insert_op_before(allocation, call)
    call.operands = [*call.arguments, allocation.memref]
    borrowed.function_type = builtin.FunctionType.from_lists(
        [*borrowed.function_type.inputs, allocation.memref.type], []
    )
    reports[0]["private_workspaces"] = [asdict(contract)]
    pool_private_writer_workspaces(module, reports)
    return module, plan, calls, reports


def test_actual_workspace_pool_insertion_remains_admitted():
    module, plan, calls, reports = pooled_fixture()
    assert install(module, plan, reports)["consumers"] == 4
    module.verify()


@pytest.mark.parametrize("change", ["extent", "extra_use", "unproved_allocation"])
def test_pool_contract_changes_refuse_transactionally(change):
    from merlin.xdsl_dialects._common import text

    module, plan, calls, reports = pooled_fixture()
    if change == "extent":
        reports[0]["private_workspaces"][0]["bytes"] += 1
    elif change == "extra_use":
        value = calls[0].arguments[-1]
        module.body.block.add_op(func.FuncOp("escape_pool", ([value.type], []), Region(), visibility="private"))
        plan.owner.body.block.insert_op_before(func.CallOp("escape_pool", [value], []), calls[2])
    else:
        extra = memref.AllocOp.get(builtin.i8, shape=[37], alignment=64)
        plan.owner.body.block.insert_op_before(extra, calls[0])
    before = text(module)
    with pytest.raises(ValueError):
        install(module, plan, reports)
    assert text(module) == before
