"""Bounded lookup preserves source table bytes through actual lowering."""

from __future__ import annotations

import subprocess

import numpy as np
import pytest
from xdsl.dialects import builtin, func
from xdsl.ir import Block, Region

from merlin.llvmlower import toolchain
from merlin.llvmlower.position_table import build_position_table_lookup


def _args(position_type=None, table_type=None):
    return Block(
        arg_types=[
            position_type or builtin.TensorType(builtin.i64, [2, 4]),
            table_type or builtin.TensorType(builtin.f32, [8, 7]),
        ]
    ).args


@pytest.mark.parametrize("bounds,minimum", [((-3, 5), -2), ((-2, 6), -2), ((5, -2), -2), ((-2, 5), 0)])
def test_refuse_uncovered_interval(bounds, minimum):
    with pytest.raises(ValueError, match="cover"):
        build_position_table_lookup(*_args(), position_bounds=bounds, table_min=minimum)


@pytest.mark.parametrize(
    "position_type,table_type",
    [
        (builtin.TensorType(builtin.i32, [2, 4]), None),
        (None, builtin.TensorType(builtin.f64, [8, 7])),
    ],
)
def test_refuse_wrong_element_type(position_type, table_type):
    with pytest.raises(TypeError):
        build_position_table_lookup(*_args(position_type, table_type), position_bounds=(-2, 5), table_min=-2)


@pytest.mark.parametrize(
    "position_type,table_type",
    [
        (builtin.TensorType(builtin.i64, [8]), None),
        (builtin.TensorType(builtin.i64, [0, 4]), None),
        (None, builtin.TensorType(builtin.f32, [-1, 7])),
    ],
)
def test_refuse_nonstatic_rank_two(position_type, table_type):
    with pytest.raises(ValueError, match="static positive rank-two"):
        build_position_table_lookup(*_args(position_type, table_type), position_bounds=(-2, 5), table_min=-2)


@pytest.mark.parametrize("bounds,minimum", [((False, 5), -2), ((-2, 5), -2.0)])
def test_require_integer_domain(bounds, minimum):
    with pytest.raises(TypeError, match="integer position interval"):
        build_position_table_lookup(*_args(), position_bounds=bounds, table_min=minimum)


def test_refuse_out_of_i64_domain():
    with pytest.raises(ValueError, match="signed i64"):
        build_position_table_lookup(*_args(), position_bounds=(1 << 63, (1 << 63) + 7), table_min=1 << 63)


@pytest.mark.parametrize("optimization", ["-O0", "-O2"])
def test_native_all_rows_and_float_bits(tmp_path, optimization):
    if not toolchain.available():
        pytest.skip("native toolchain unavailable")
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.lower import lower_model_file

    block = Block(arg_types=[builtin.TensorType(builtin.i64, [2, 4]), builtin.TensorType(builtin.f32, [8, 7])])
    ops, result = build_position_table_lookup(*block.args, position_bounds=(-2, 5), table_min=-2)
    block.add_ops([*ops, func.ReturnOp(result)])
    module = builtin.ModuleOp([func.FuncOp("forward", ([a.type for a in block.args], [result.type]), Region(block))])
    module.verify()
    source = tmp_path / "input.mlir"
    source.write_text(str(module))
    lowered = lower_model_file(source, tmp_path / "lower", targets=(), textual=True)
    obj = tmp_path / "model.o"
    shared = tmp_path / f"model{optimization}.so"
    subprocess.run(
        [str(toolchain.clang()), optimization, "-fPIC", "-c", str(lowered.ll_path), "-o", str(obj)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["cc", "-shared", "-fPIC", str(obj), str(mlir_runtime_c()), "-o", str(shared)], check=True, capture_output=True
    )
    positions = np.array([[-2, 5, 0, 3], [1, 4, -1, 2]], dtype=np.int64)
    bits = np.arange(56, dtype=np.uint32).reshape(8, 7) + 0x3F800000
    bits[0, :4] = [0, 0x80000000, 0x7FC01234, 0xFF800000]
    table = bits.view(np.float32)
    actual = np.zeros((2, 4, 1, 7), dtype=np.float32)
    HostModel.load(str(shared))([(a.ctypes.data, a.shape) for a in [positions, table, actual]])
    expected = np.stack([table[int(p) + 2] for p in positions.flat]).reshape(actual.shape)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
