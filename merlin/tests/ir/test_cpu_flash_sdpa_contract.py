"""Admission and arithmetic structure of the opt-in CPU flash schedule."""

import pytest
from xdsl.dialects import arith, builtin, func, math
from xdsl.dialects.linalg import ops as linalg
from xdsl.ir import Block, Region

from merlin.llvmlower.cpu_flash_exp import CPU_FLASH_BF16_AVX2_POLICY
from merlin.llvmlower.cpu_flash_sdpa import CPU_FLASH_PV_MKL_DEF_ZEN_POLICY, build_cpu_flash_sdpa


def _args(shapes=((1, 2, 768, 64), (1, 2, 1024, 64), (1, 2, 1024, 64)), dtype=builtin.bf16):
    return Block(arg_types=[builtin.TensorType(dtype, shape) for shape in shapes])


def _build(block, **overrides):
    options = dict(backend_policy=CPU_FLASH_BF16_AVX2_POLICY, scale=0.125, assume_finite_intermediates=True)
    options.update(overrides)
    return build_cpu_flash_sdpa(*block.args, **options)


@pytest.mark.parametrize(
    "options",
    [
        {"backend_policy": "math"},
        {"pv_backend_policy": "unproven_gemm"},
        {"assume_finite_intermediates": False},
        {"dropout_p": 0.1},
        {"is_causal": True},
        {"enable_gqa": True},
        {"scale": 0},
        {"scale": -0.1},
        {"scale": float("inf")},
        {"scale": float("nan")},
        {"scale": 1e300},
        {"scale": 1e-300},
    ],
)
def test_refuses_unadmitted_policy_and_options_without_mutation(options):
    block = _args()
    with pytest.raises(ValueError):
        _build(block, **options)
    assert not list(block.ops)
    assert all(not tuple(arg.uses) for arg in block.args)


@pytest.mark.parametrize(
    "shapes",
    [
        ((1, 2, 512, 64), (1, 2, 1024, 64), (1, 2, 1024, 64)),
        ((1, 2, 769, 64), (1, 2, 1024, 64), (1, 2, 1024, 64)),
        ((1, 2, 768, 64), (1, 2, 513, 64), (1, 2, 513, 64)),
        ((1, 2, 768, 32), (1, 2, 1024, 32), (1, 2, 1024, 32)),
        ((1, 2, 768, 64), (1, 1, 1024, 64), (1, 1, 1024, 64)),
        ((1, 2, 768, 64), (1, 2, 1024, 64), (1, 2, 512, 64)),
        ((1, 2, -1, 64), (1, 2, 1024, 64), (1, 2, 1024, 64)),
        ((2, 768, 64), (2, 1024, 64), (2, 1024, 64)),
    ],
)
def test_refuses_shapes_outside_audited_domain(shapes):
    with pytest.raises(ValueError):
        _build(_args(shapes))


@pytest.mark.parametrize("dtype", [builtin.f16, builtin.f32, builtin.i8])
def test_refuses_other_input_types(dtype):
    with pytest.raises(TypeError):
        _build(_args(dtype=dtype))


def test_verifies_bf16_output_and_source_arithmetic_structure():
    block = _args()
    ops, result = _build(block)
    assert result.type == block.args[0].type
    block.add_ops([*ops, func.ReturnOp(result)])
    module = builtin.ModuleOp(
        [func.FuncOp("attention", ([arg.type for arg in block.args], [result.type]), Region(block))]
    )
    module.verify()
    assert all(op.attributes["merlin.sdpa_backend_policy"].data == CPU_FLASH_BF16_AVX2_POLICY for op in ops)
    # Each query tile has two512-key blocks. Exponential sums must consume f32
    # values before bf16 probability narrowing. The source eight-lane reduction
    # retains64 groups and reduces that group dimension, never all512 serially.
    lane_sums = [
        op for op in module.walk() if isinstance(op, linalg.ReduceOp) and len(op.operands[0].type.get_shape()) == 5
    ]
    assert len(lane_sums) == 6
    for op in lane_sums:
        assert op.operands[0].type.get_shape()[-2:] == (64, 8)
        assert op.operands[0].type.element_type == builtin.f32
        assert tuple(op.dimensions.iter_values()) == (3,)
    contractions = [op for op in module.walk() if isinstance(op, linalg.GenericOp) and len(op.iterator_types) == 5]
    assert len(contractions) == 12
    for op in contractions:
        assert [arg.type for arg in op.body.block.args] == [builtin.bf16, builtin.bf16, builtin.f32]
        arithmetic = [nested.name for nested in op.body.block.ops]
        assert arithmetic == ["arith.extf", "arith.extf", "math.fma", "linalg.yield"]
    # The explicit exponential builder uses4 FMAs per key block; each block
    # additionally has two contraction FMAs and one denominator FMA.
    assert sum(isinstance(op, math.FmaOp) for op in module.walk()) == 6 * 7
    assert sum(isinstance(op, math.ExpOp) for op in module.walk()) == 6
    assert sum(isinstance(op, arith.DivfOp) for op in module.walk()) == 3
    assert sum(isinstance(op, arith.TruncFOp) for op in module.walk()) == 6 + 3
    for op in module.walk():
        flags = op.properties.get("fastmath")
        assert flags is None or not flags.data


def test_explicit_mkl_def_zen_panels_have_zero_seed_and_separate_destination_add():
    block = _args()
    ops, result = _build(block, pv_backend_policy=CPU_FLASH_PV_MKL_DEF_ZEN_POLICY)
    block.add_ops([*ops, func.ReturnOp(result)])
    module = builtin.ModuleOp(
        [func.FuncOp("attention", ([arg.type for arg in block.args], [result.type]), Region(block))]
    )
    module.verify()
    contractions = [op for op in ops if isinstance(op, linalg.GenericOp) and len(op.iterator_types) == 5]
    assert len(contractions) == 24
    for offset in range(0, len(contractions), 4):
        panels = contractions[offset + 1 : offset + 4]
        assert [op.inputs[0].type.get_shape()[-1] for op in panels] == [192, 192, 128]
        for panel in panels:
            splat = panel.outputs[0].owner
            assert splat.name == "tensor.splat"
            assert splat.operands[0].owner.value.value.data == 0.0
            uses = tuple(panel.results[0].uses)
            assert len(uses) == 1
            assert isinstance(uses[0].operation.body.block.first_op, arith.AddfOp)
    assert all(op.attributes["merlin.sdpa_pv_backend_policy"].data == CPU_FLASH_PV_MKL_DEF_ZEN_POLICY for op in ops)
