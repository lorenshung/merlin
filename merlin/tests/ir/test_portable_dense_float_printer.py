"""Executable dense float constants preserve bits through both IR readers."""

import subprocess
from pathlib import Path

import pytest
from xdsl.dialects import arith, builtin, func
from xdsl.ir import Block, Region

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.toolchain import m2m_python
from merlin.xdsl_dialects._common import text


@pytest.mark.parametrize(
    "typ,bits",
    [
        (builtin.f16, [0, 0x8000, 0x7C00, 0xFC00, 0x7E21, 1, 0x3C00]),
        (builtin.bf16, [0, 0x8000, 0x7F80, 0xFF80, 0x7FC1, 1, 0x3F80]),
        (builtin.f32, [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC01234, 1, 0x4B400000]),
        (
            builtin.f64,
            [0, 0x8000000000000000, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FF8000000001234, 1, 0x3FF0000000000000],
        ),
    ],
)
@pytest.mark.parametrize("splat", [False, True])
def test_dense_float_bytes_roundtrip(typ, bits, splat):
    if splat:
        bits = [bits[3]] * 7
    raw = b"".join(x.to_bytes(typ.compile_time_size, "little") for x in bits)
    value = builtin.DenseIntOrFPElementsAttr(builtin.TensorType(typ, [len(bits)]), builtin.BytesAttr(raw))
    module = builtin.ModuleOp([arith.ConstantOp(value)])
    for _ in range(3):
        printed = text(module)
        assert 'dense<"0x' in printed
        module = parse_mlir_text(printed)
        module.verify()
        assert module.body.block.first_op.value.data.data == raw
    proc = subprocess.run(
        [
            str(m2m_python()),
            "-c",
            "import sys; from torch_mlir import ir; "
            "ctx=ir.Context(); module=ir.Module.parse(sys.stdin.read(),ctx); module.operation.verify()",
        ],
        input=printed,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_dense_float_native_keeps_inf_nan_and_signed_zero(tmp_path):
    import numpy as np

    from merlin.llvmlower import toolchain
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.lower import lower_model_file

    bits = np.array([0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC01234, 1, 0x4B400000], dtype=np.uint32)
    typ = builtin.TensorType(builtin.f32, [len(bits)])
    value = arith.ConstantOp(builtin.DenseIntOrFPElementsAttr(typ, builtin.BytesAttr(bits.tobytes())))
    block = Block([value, func.ReturnOp(value.result)])
    module = builtin.ModuleOp([func.FuncOp("forward", ([], [typ]), Region(block))])
    # Exercise the same repeated xDSL serialization as model preparation.
    source = tmp_path / "input.mlir"
    source.write_text(text(parse_mlir_text(text(module))))
    lowered = lower_model_file(source, tmp_path / "lower", targets=(), textual=True)
    obj = tmp_path / "model.o"
    lib = tmp_path / "model.so"
    subprocess.run([str(toolchain.clang()), "-O2", "-fPIC", "-c", str(lowered.ll_path), "-o", str(obj)], check=True)
    subprocess.run(["cc", "-shared", "-fPIC", str(obj), str(mlir_runtime_c()), "-lm", "-o", str(lib)], check=True)
    result = np.zeros(bits.shape, dtype=np.float32)
    HostModel.load(str(lib))([(result.ctypes.data, result.shape)])
    np.testing.assert_array_equal(result.view(np.uint32), bits)
