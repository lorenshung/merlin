"""Boolean CPU flash masks, including zero-probability online-softmax rows."""

import ctypes
import shutil
import subprocess

import numpy as np
import pytest
from xdsl.dialects import arith, builtin, func, tensor
from xdsl.dialects.linalg import ops as linalg
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap

from merlin.llvmlower import toolchain
from merlin.llvmlower.cpu_flash_exp import CPU_FLASH_BF16_AVX2_POLICY
from merlin.llvmlower.cpu_flash_sdpa import CPU_FLASH_PV_MKL_DEF_ZEN_POLICY, build_cpu_flash_sdpa


def _build(block):
    return build_cpu_flash_sdpa(
        *block.args[:3],
        mask=block.args[3],
        backend_policy=CPU_FLASH_BF16_AVX2_POLICY,
        scale=0.125,
        assume_finite_intermediates=True,
    )


@pytest.mark.parametrize(
    "mask_type,error",
    [
        (builtin.TensorType(builtin.f32, [1, 1, 768, 1024]), TypeError),
        (builtin.TensorType(builtin.i8, [1, 1, 768, 1024]), TypeError),
        (builtin.TensorType(builtin.i1, [768, 1024]), ValueError),
        (builtin.TensorType(builtin.i1, [1, 1, 769, 1024]), ValueError),
        (builtin.TensorType(builtin.i1, [1, 1, 768, 512]), ValueError),
        (builtin.TensorType(builtin.i1, [-1, 1, 768, 1024]), ValueError),
    ],
)
def test_refuse_unadmitted_mask_types_and_shapes(mask_type, error):
    block = Block(arg_types=[builtin.TensorType(builtin.bf16, [1, 1, n, 64]) for n in (768, 1024, 1024)] + [mask_type])
    with pytest.raises(error):
        _build(block)
    assert all(not tuple(arg.uses) for arg in block.args)


@pytest.mark.parametrize("mask_shape", [(1, 1, 1, 1), (1, 2, 768, 1), (2, 1, 1, 1024), (2, 2, 768, 1024)])
def test_accept_and_verify_broadcast_masks(mask_shape):
    types = [builtin.TensorType(builtin.bf16, [2, 2, n, 64]) for n in (768, 1024, 1024)]
    types.append(builtin.TensorType(builtin.i1, mask_shape))
    block = Block(arg_types=types)
    ops, result = _build(block)
    block.add_ops([*ops, func.ReturnOp(result)])
    module = builtin.ModuleOp([func.FuncOp("forward", (types, [result.type]), Region(block))])
    module.verify()


def _native_module(mask_shape, pv_backend_policy=None):
    shapes = ([1, 1, 768, 64], [1, 1, 1024, 64], [1, 1, 1024, 64])
    types = [builtin.TensorType(builtin.f32, shape) for shape in shapes]
    if mask_shape is not None:
        types.append(builtin.TensorType(builtin.i1, mask_shape))
    block = Block(arg_types=types)

    def cast(value, dtype, op_type):
        result_type = builtin.TensorType(dtype, value.type.get_shape())
        empty = tensor.EmptyOp((), result_type)
        body = Block(arg_types=[value.type.element_type, dtype])
        operation = op_type(body.args[0], dtype)
        body.add_ops([operation, linalg.YieldOp(operation.result)])
        generic = linalg.GenericOp(
            inputs=(value,),
            outputs=(empty.tensor,),
            body=Region(body),
            indexing_maps=builtin.ArrayAttr([builtin.AffineMapAttr(AffineMap.identity(4))] * 2),
            iterator_types=builtin.ArrayAttr([linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL)] * 4),
            result_types=(result_type,),
        )
        block.add_ops([empty, generic])
        return generic.results[0]

    operands = [cast(arg, builtin.bf16, arith.TruncFOp) for arg in block.args[:3]]
    ops, result = build_cpu_flash_sdpa(
        *operands,
        mask=block.args[3] if mask_shape is not None else None,
        backend_policy=CPU_FLASH_BF16_AVX2_POLICY,
        scale=0.125,
        assume_finite_intermediates=True,
        pv_backend_policy=pv_backend_policy,
    )
    block.add_ops(ops)
    result = cast(result, builtin.f32, arith.ExtFOp)
    block.add_op(func.ReturnOp(result))
    module = builtin.ModuleOp([func.FuncOp("forward", (types, [result.type]), Region(block))])
    module.verify()
    return module


@pytest.mark.parametrize("optimization", ["-O0", "-O2"])
def test_native_mkl_panel_order_cancellation(tmp_path, optimization):
    if not toolchain.available() or not shutil.which(str(toolchain.mlir_translate())):
        pytest.skip("native panel test requires LLVM tools")
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.lower import lower_model_file

    source = tmp_path / "panels.mlir"
    source.write_text(str(_native_module(None, CPU_FLASH_PV_MKL_DEF_ZEN_POLICY)))
    lowered = lower_model_file(
        source, tmp_path / "lower", targets=(), textual=True, features=frozenset({"lower_fma_to_intrinsic"})
    )
    obj, shared = tmp_path / "panels.o", tmp_path / "panels.so"
    subprocess.run(
        [str(toolchain.clang()), optimization, "-fPIC", "-c", str(lowered.ll_path), "-o", str(obj)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["cc", "-shared", "-fPIC", str(obj), str(mlir_runtime_c()), "-lm", "-o", str(shared)],
        check=True,
        capture_output=True,
    )
    q = np.zeros((1, 1, 768, 64), dtype=np.float32)
    k = np.zeros((1, 1, 1024, 64), dtype=np.float32)
    v = np.zeros_like(k)
    v[:, :, 0] = 2**25
    v[:, :, 192] = -(2**25)
    v[:, :, 193] = 1
    v[:, :, 512] = 0.5
    # Each panel starts zero: the second panel loses its +1 before cancellation
    # with the first panel. A serial seeded reduction instead retains that +1.
    expected = np.full_like(q, 0.5 / 1024)
    output = np.empty_like(q)
    HostModel.load(str(shared))([(a.ctypes.data, a.shape) for a in (q, k, v, output)])
    np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize("optimization", ["-O0", "-O2"])
@pytest.mark.parametrize("scenario", ["rows", "broadcast_query", "unmasked"])
def test_native_masked_rows_and_first_key_tile(tmp_path, scenario, optimization):
    if not toolchain.available() or not shutil.which(str(toolchain.mlir_translate())):
        pytest.skip("native mask tests require compiler Python, clang and mlir-translate")
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.lower import lower_model_file

    # pytest may recycle a deleted passing-test directory; distinct paths keep
    # dlopen from returning the preceding case's cached shared library.
    tmp_path = tmp_path / (scenario + optimization)
    tmp_path.mkdir()
    mask_shape = None if scenario == "unmasked" else [1, 1, 1 if scenario == "broadcast_query" else 768, 1024]
    source = tmp_path / "masked.mlir"
    source.write_text(str(_native_module(mask_shape)))
    lowered = lower_model_file(source, tmp_path / "lower", targets=(), textual=True)
    obj, shared = tmp_path / "masked.o", tmp_path / "masked.so"
    subprocess.run(
        [str(toolchain.clang()), optimization, "-fPIC", "-c", str(lowered.ll_path), "-o", str(obj)],
        check=True,
        capture_output=True,
        text=True,
    )
    fenv = tmp_path / "fenv.c"
    fenv.write_text(
        "#include <fenv.h>\n"
        "void clear_flags(void){feclearexcept(FE_ALL_EXCEPT); }\n"
        "int invalid_flag(void){return fetestexcept(FE_INVALID); }\n"
    )
    subprocess.run(
        ["cc", "-shared", "-fPIC", str(obj), str(mlir_runtime_c()), str(fenv), "-lm", "-o", str(shared)],
        check=True,
        capture_output=True,
        text=True,
    )
    q = np.zeros((1, 1, 768, 64), dtype=np.float32)
    k = np.zeros((1, 1, 1024, 64), dtype=np.float32)
    v = np.empty_like(k)
    v[:, :, :512] = 2
    v[:, :, 512:] = 4
    mask = np.ones(mask_shape, dtype=np.bool_) if mask_shape is not None else None
    expected = np.full_like(q, 3)
    if scenario == "broadcast_query":
        mask[:, :, :, :512] = False
        expected.fill(4)
    elif scenario == "rows":
        mask[:, :, 0, :] = False  # Whole row masked: output must be zero, not NaN.
        mask[:, :, 1, :512] = False  # First key tile masked, next tile becomes valid.
        mask[:, :, 2, 512:] = False  # A masked later tile preserves the previous result.
        expected[:, :, 0] = 0
        expected[:, :, 1] = 4
        expected[:, :, 2] = 2
    output = np.full_like(q, np.nan)
    arrays = [q, k, v] + ([mask] if mask is not None else []) + [output]
    host = HostModel.load(str(shared))
    flags = ctypes.CDLL(str(shared))
    flags.clear_flags()
    host([(a.ctypes.data, a.shape) for a in arrays])
    invalid = flags.invalid_flag()
    # At O0 verify the emitted safe operands avoid invalid subtraction. LLVM
    # unconstrained FP can speculate at O2; exception flags are not promised.
    if optimization == "-O0":
        assert invalid == 0
    # Equal logits and exactly representable constant values make the source
    # bf16 result exact despite backend differences in contraction summation.
    np.testing.assert_array_equal(output, expected)
    assert np.isfinite(output).all()


def test_pinned_cpu_flash_source_mask_fixture():
    torch = pytest.importorskip("torch")
    q = torch.zeros(1, 1, 768, 64, dtype=torch.bfloat16)
    k = torch.zeros(1, 1, 1024, 64, dtype=torch.bfloat16)
    v = torch.empty_like(k)
    v[:, :, :512] = 2
    v[:, :, 512:] = 4
    mask = torch.zeros(1, 1, 768, 1024, dtype=torch.float32)
    mask[:, :, 0] = -float("inf")
    mask[:, :, 1, :512] = -float("inf")
    mask[:, :, 2, 512:] = -float("inf")
    expected = torch.full_like(q, 3)
    expected[:, :, 0] = 0
    expected[:, :, 1] = 4
    expected[:, :, 2] = 2
    actual, _ = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(
        q,
        k,
        v,
        attn_mask=mask,
        scale=0.125,
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
