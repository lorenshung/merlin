"""Source partition keeps arithmetic and caller boundaries through actual lowering."""

import hashlib
import subprocess

import numpy as np
import pytest
from xdsl.dialects import builtin, func

from merlin.llvmlower.abi import HostModel
from merlin.llvmlower.codegen import mlir_runtime_c
from merlin.llvmlower.ordered_fma_group_outline import outline_ordered_fma_group
from merlin.llvmlower.ordered_fma_groups import analyze_ordered_fma_groups
from merlin.llvmlower.pipeline import lower_to_llvm_ir
from merlin.llvmlower.toolchain import clang
from merlin.tests.ir.test_ordered_fma_groups import simple
from merlin.xdsl_dialects._common import text


def test_moves_original_operations_and_preserves_exact_zero_source():
    module, root, scaled, endpoint = simple()
    group = analyze_ordered_fma_groups(module)[0]
    original = tuple(group.operations)
    outlined = outline_ordered_fma_group(module, group, "closed_source")
    assert outlined.source_operations == original
    assert all(operation.parent is outlined.function.body.block for operation in original)
    assert len(outlined.inputs) == 2
    assert len(outlined.omitted_unread_initializers) == 2
    assert len(outlined.internalized_constants) == 1
    assert tuple(outlined.call.arguments) == outlined.inputs
    assert module.body.block.first_op.body.block.last_op.arguments[0] is outlined.call.results[0]
    closed = analyze_ordered_fma_groups(module)[0]
    assert closed.closed_bf16_endpoints and closed.contractions == (root,)


@pytest.mark.parametrize("mutation", ["float_escape", "symbol_collision", "strict_context"])
def test_refusal_is_transactional(mutation):
    module, root, scaled, endpoint = simple(mutation == "float_escape")
    group = analyze_ordered_fma_groups(module)[0]
    if mutation == "symbol_collision":
        module.body.block.add_op(func.FuncOp.external("closed_source", [], []))
    elif mutation == "strict_context":
        module.body.block.first_op.attributes["strictfp"] = builtin.UnitAttr()
    before = text(module)
    with pytest.raises(ValueError):
        outline_ordered_fma_group(module, group, "closed_source")
    assert text(module) == before


def compile_and_run(tmp_path, source, arguments):
    work = tmp_path / hashlib.sha256(source.encode()).hexdigest()[:12]
    work.mkdir()
    llvm = lower_to_llvm_ir(source, workdir=work, features={"lower_fma_to_intrinsic"})
    path = work / "model.ll"
    path.write_text(llvm)
    library = work / "model.so"
    subprocess.run(
        [
            str(clang()),
            "-O2",
            "-march=native",
            "-fPIC",
            "-shared",
            "-ffp-contract=off",
            str(path),
            str(mlir_runtime_c()),
            "-lm",
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
    )
    output = np.empty((2, 3, 5), dtype="u2")
    HostModel.load(str(library))(
        [(argument.ctypes.data, argument.shape) for argument in arguments] + [(output.ctypes.data, output.shape)]
    )
    return output


@pytest.mark.parametrize("values", ["independent", "cancellation", "special"])
def test_actual_compiled_source_partition_matches_original_bits(tmp_path, values):
    module, root, scaled, endpoint = simple()
    module.body.block.first_op.attributes["llvm.emit_c_interface"] = builtin.UnitAttr()
    original = text(module)
    outline_ordered_fma_group(module, analyze_ordered_fma_groups(module)[0], "closed_source")
    transformed = text(module)
    rng = np.random.default_rng(832)
    a = rng.normal(size=(2, 3, 4)).astype("f4")
    b = rng.normal(size=(2, 4, 5)).astype("f4")
    if values == "cancellation":
        a[:] = np.array([1.0, 1.0, -1.0, -1.0], "f4")
        b[:] = 1.0
        b[:, -1, -1] = np.float32(1.00390625)
    elif values == "special":
        a[0, 0] = np.array([0.0, -0.0, np.inf, 1.0], "f4")
        b[0, -1, -1] = np.nan
    # Independent BF16 inputs; preserve literal bit patterns including zeros/NaN.
    arguments = [(argument.view("u4") >> 16).astype("u2") for argument in (a, b)]
    control = compile_and_run(tmp_path, original, arguments)
    candidate = compile_and_run(tmp_path, transformed, arguments)
    np.testing.assert_array_equal(candidate, control)
