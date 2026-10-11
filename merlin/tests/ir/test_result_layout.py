"""Native result descriptors preserve frontend storage semantics."""

import ctypes
import subprocess

import numpy as np
import pytest

from merlin.llvmlower.pipeline import lower_to_llvm_ir
from merlin.llvmlower.result_layout import module_attribute
from merlin.llvmlower.toolchain import clang, llvm_install, m2m_python


class Descriptor(ctypes.Structure):
    _fields_ = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
        ("sizes", ctypes.c_int64 * 2),
        ("strides", ctypes.c_int64 * 2),
    ]


def execute(tmp_path, result, *, dynamic=False, extent=4):
    if not m2m_python().is_file():
        pytest.skip("native MLIR interpreter unavailable")
    import hashlib

    tmp_path = tmp_path / hashlib.sha256(str((result, extent)).encode()).hexdigest()[:12]
    tmp_path.mkdir()
    contract = dict(inputs=[dict(shape=[extent, 5], stride=[5, 1], storage_offset=0)], results=[result])
    shape = "?x5" if dynamic else "x".join(str(n) for n in result["shape"])
    body = (
        f"%b = tensor.cast %a : tensor<{extent}x5xf32> to tensor<?x5xf32>"
        if dynamic
        else f"%b = tensor.empty() : tensor<{shape}xf32>"
    )
    source = (
        "builtin.module"
        + module_attribute(contract)
        + f""" {{
      func.func @forward(%a: tensor<{extent}x5xf32>) -> tensor<{shape}xf32>
          attributes {{llvm.emit_c_interface}} {{
        {body}
        func.return %b : tensor<{shape}xf32>
      }}
    }}"""
    )
    lowered = lower_to_llvm_ir(source, workdir=tmp_path, preserve_result_descriptors=True)
    path, library = tmp_path / "model.ll", tmp_path / "model.so"
    path.write_text(lowered)
    lib = llvm_install() / "lib"
    subprocess.run(
        [
            str(clang()),
            "-shared",
            "-fPIC",
            "-O1",
            "-Wl,-Bsymbolic",
            str(path),
            "-L" + str(lib),
            "-lmlir_c_runner_utils",
            "-Wl,-rpath," + str(lib),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
    )
    values = np.arange(extent * 5, dtype=np.float32).reshape(extent, 5)
    arg = Descriptor(
        values.ctypes.data, values.ctypes.data, 0, (ctypes.c_int64 * 2)(extent, 5), (ctypes.c_int64 * 2)(5, 1)
    )
    output = Descriptor()
    entry = ctypes.CDLL(str(library))._mlir_ciface_forward
    entry.argtypes = [ctypes.POINTER(Descriptor), ctypes.POINTER(Descriptor)]
    entry(ctypes.byref(output), ctypes.byref(arg))
    return values, output


@pytest.mark.parametrize("shape,stride,offset", [([2, 3], [5, 1], 6), ([5, 4], [1, 5], 0), ([3, 5], [0, 1], 0)])
def test_borrowed_native_view_shares_allocation(tmp_path, shape, stride, offset):
    values, output = execute(
        tmp_path, dict(shape=shape, stride=stride, storage_offset=offset, alias_inputs=[0], role="user_output")
    )
    assert output.allocated == values.ctypes.data
    assert output.aligned == values.ctypes.data
    assert output.offset == offset
    assert list(output.sizes) == shape
    assert list(output.strides) == stride
    pointer = ctypes.cast(output.aligned, ctypes.POINTER(ctypes.c_float))
    pointer[offset] = -17
    assert values.ravel()[offset] == -17  # A write through the result reaches the input.


@pytest.mark.parametrize("extent", [0, 1, 4])
def test_dynamic_fresh_result_is_physically_column_major(tmp_path, extent):
    values, output = execute(
        tmp_path,
        dict(shape=[-1, 5], stride=[1, {"shape_axes": [0]}], storage_offset=0, alias_inputs=[], role="user_output"),
        dynamic=True,
        extent=extent,
    )
    assert output.allocated != values.ctypes.data
    assert list(output.sizes) == [extent, 5]
    assert list(output.strides) == [1, extent]
    pointer = ctypes.cast(output.aligned, ctypes.POINTER(ctypes.c_float))
    actual = [[pointer[output.offset + i + extent * j] for j in range(5)] for i in range(extent)]
    np.testing.assert_array_equal(np.asarray(actual).reshape(extent, 5), values)
    ctypes.CDLL(None).free(ctypes.c_void_p(output.allocated))


def test_fresh_alias_class_covers_later_full_storage(tmp_path):
    if not m2m_python().is_file():
        pytest.skip("native MLIR interpreter unavailable")

    class Pair(ctypes.Structure):
        _fields_ = [("partial", Descriptor), ("full", Descriptor)]

    contract = dict(
        inputs=[dict(shape=[4, 5], stride=[5, 1], storage_offset=0)],
        results=[
            dict(shape=[2, 5], stride=[5, 1], storage_offset=5, alias_inputs=[]),
            dict(shape=[4, 5], stride=[5, 1], storage_offset=0, alias_inputs=[], alias_results=[0]),
        ],
    )
    source = (
        "builtin.module"
        + module_attribute(contract)
        + """ {
      func.func @forward(%a: tensor<4x5xf32>) -> (tensor<2x5xf32>, tensor<4x5xf32>)
          attributes {llvm.emit_c_interface} {
        %b = "tensor.extract_slice"(%a) <{static_offsets = array<i64: 1, 0>,
          static_sizes = array<i64: 2, 5>, static_strides = array<i64: 1, 1>,
          operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<4x5xf32>) -> tensor<2x5xf32>
        func.return %b, %a : tensor<2x5xf32>, tensor<4x5xf32>
      }
    }"""
    )
    lowered = lower_to_llvm_ir(source, workdir=tmp_path, preserve_result_descriptors=True)
    path, library = tmp_path / "pair.ll", tmp_path / "pair.so"
    path.write_text(lowered)
    lib = llvm_install() / "lib"
    subprocess.run(
        [
            str(clang()),
            "-shared",
            "-fPIC",
            "-O1",
            "-Wl,-Bsymbolic",
            str(path),
            "-L" + str(lib),
            "-lmlir_c_runner_utils",
            "-Wl,-rpath," + str(lib),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
    )
    values = np.arange(20, dtype=np.float32)
    arg = Descriptor(values.ctypes.data, values.ctypes.data, 0, (ctypes.c_int64 * 2)(4, 5), (ctypes.c_int64 * 2)(5, 1))
    output = Pair()
    entry = ctypes.CDLL(str(library))._mlir_ciface_forward
    entry.argtypes = [ctypes.POINTER(Pair), ctypes.POINTER(Descriptor)]
    entry(ctypes.byref(output), ctypes.byref(arg))
    assert output.partial.allocated == output.full.allocated != values.ctypes.data
    assert output.partial.offset == 5 and output.full.offset == 0
    pointer = ctypes.cast(output.full.aligned, ctypes.POINTER(ctypes.c_float))
    np.testing.assert_array_equal([pointer[i] for i in range(20)], values)
    pointer[output.partial.offset] = -31
    assert ctypes.cast(output.partial.aligned, ctypes.POINTER(ctypes.c_float))[5] == -31
    ctypes.CDLL(None).free(ctypes.c_void_p(output.full.allocated))


def apply_memref_layout(source, contract, *, operation_owners=False):
    """Exercise the ownership rewrite in its native compiler interpreter."""
    import json

    code = """
import json, sys
from torch_mlir import ir
from torch_mlir.dialects import arith, memref
from merlin.llvmlower.result_layout import _apply_result_layout
source, contract, operation_owners = json.load(sys.stdin)
# Some compiler bindings expose bare Operation owners. Exercise that branch
# alongside native OpView owners without changing the parsed IR or its values.
if operation_owners:
    ir.Operation = (ir.Operation, ir.OpView)
with ir.Context() as ctx:
    module = ir.Module.parse(source)
    _apply_result_layout(module, contract)
    print(module)
"""
    return subprocess.run(
        [str(m2m_python()), "-c", code],
        input=json.dumps([source, contract, operation_owners]),
        text=True,
        capture_output=True,
    )


@pytest.mark.parametrize("operation_owners", [False, True])
@pytest.mark.parametrize("producer", ["transpose", "view", "alloc", "argument"])
def test_discarded_alloc_is_retained_when_live_root_is_unknown(producer, operation_owners):
    if producer == "view":
        input_type = "memref<24xi8>"
        live_type = "memref<2x3xf32>"
        body = (
            "%zero = arith.constant 0 : index\n%live = memref.view %owned[%zero][] : memref<24xi8> to memref<2x3xf32>"
        )
        input_shape, input_stride = [24], [1]
    else:
        input_type = "memref<2x3xf32>"
        input_shape, input_stride = [2, 3], [3, 1]
        if producer == "transpose":
            live_type = "memref<3x2xf32, strided<[1, 3]>>"
            body = "%live = memref.transpose %owned (d0, d1) -> (d1, d0) : memref<2x3xf32> to " + live_type
        elif producer == "alloc":
            live_type = input_type
            body = "%live = memref.alloc() : " + live_type
        else:
            live_type = input_type
            body = "%live = memref.cast %arg : memref<2x3xf32> to memref<2x3xf32>"
    source = f"""module {{
      func.func @forward(%arg: {input_type}) -> ({input_type}, {live_type}) {{
        %owned = memref.alloc() : {input_type}
        {body}
        return %owned, %live : {input_type}, {live_type}
      }}
    }}"""
    record = dict(shape=input_shape, stride=input_stride, storage_offset=0)
    result = apply_memref_layout(
        source, dict(inputs=[record], results=[dict(record, alias_inputs=[0]), None]), operation_owners=operation_owners
    )
    assert result.returncode == 0, result.stderr
    assert ("memref.dealloc" in result.stdout) == (producer in {"alloc", "argument"})
    # The original owned allocation remains the backing storage of the unknown view.
    if producer in {"transpose", "view"}:
        assert "memref.alloc" in result.stdout and "memref." + producer in result.stdout


@pytest.mark.parametrize("unknown", ["shape", "stride", "offset", "capacity", "compiled_result", "compiled_input"])
def test_borrowed_view_refuses_unproven_span(unknown):
    input_record = dict(shape=[2, 3], stride=[3, 1], storage_offset=0)
    record = dict(shape=[2, 3], stride=[3, 1], storage_offset=0, alias_inputs=[0])
    input_type = "memref<2x3xf32>"
    result_type = "memref<2x3xf32>"
    if unknown == "shape":
        record["shape"] = [-1, 3]
        result_type = "memref<?x3xf32>"
    elif unknown == "stride":
        record["stride"] = [{"shape_axes": [1]}, 1]
    elif unknown == "offset":
        record["storage_offset"] = {"shape_axes": [0]}
    elif unknown == "capacity":
        input_record["shape"] = [-1, 3]
        input_type = "memref<?x3xf32>"
    elif unknown == "compiled_result":
        result_type = "memref<?x3xf32>"
    else:
        input_type = "memref<?x3xf32>"
    source = f"""module {{
      func.func @forward(%arg: {input_type}, %value: {result_type}) -> {result_type} {{
        return %value : {result_type}
      }}
    }}"""
    other_input = dict(shape=[2, 3], stride=[3, 1], storage_offset=0)
    result = apply_memref_layout(source, dict(inputs=[input_record, other_input], results=[record]))
    assert result.returncode != 0
    reason = "cannot prove input storage injectivity" if unknown == "capacity" else "cannot prove returned view fits"
    assert reason in result.stderr


def test_borrowed_view_refuses_static_out_of_bounds_span():
    source = """module {
      func.func @forward(%arg: memref<2x3xf32>, %value: memref<2x3xf32>) -> memref<2x3xf32> {
        return %value : memref<2x3xf32>
      }
    }"""
    inputs = [dict(shape=[2, 3], stride=[3, 1], storage_offset=0)] * 2
    record = dict(shape=[2, 3], stride=[3, 1], storage_offset=1, alias_inputs=[0])
    result = apply_memref_layout(source, dict(inputs=inputs, results=[record]))
    assert result.returncode != 0
    assert "exceeds its input storage envelope" in result.stderr


@pytest.mark.parametrize("dynamic", [True, False])
def test_view_of_mutation_result_requires_its_own_capacity_proof(dynamic):
    shape, result_type = ([-1, 3], "memref<?x3xf32>") if dynamic else ([4, 3], "memref<4x3xf32>")
    source = f"""module {{
      func.func @forward(%arg: memref<2x3xf32>, %value: {result_type}) -> (memref<2x3xf32>, {result_type}) {{
        return %arg, %value : memref<2x3xf32>, {result_type}
      }}
    }}"""
    original = dict(shape=[2, 3], stride=[3, 1], storage_offset=0)
    mutation = dict(original, alias_inputs=[0], role="user_input_mutation", input_index=0)
    view = dict(shape=shape, stride=[3, 1], storage_offset=0, alias_inputs=[0], role="user_output")
    result = apply_memref_layout(source, dict(inputs=[original, dict(original, shape=shape)], results=[mutation, view]))
    assert result.returncode != 0
    assert ("cannot prove returned view fits" if dynamic else "exceeds its input storage envelope") in result.stderr


def test_dynamic_mutation_cannot_supply_an_unproven_borrowed_view():
    source = """module {
      func.func @forward(%arg: memref<2x3xf32>, %value: memref<?x3xf32>) -> (memref<?x3xf32>, memref<2x3xf32>) {
        return %value, %arg : memref<?x3xf32>, memref<2x3xf32>
      }
    }"""
    record = dict(shape=[2, 3], stride=[3, 1], storage_offset=0)
    mutation = dict(record, alias_inputs=[0], role="user_input_mutation", input_index=0)
    view = dict(record, alias_inputs=[0], role="user_output")
    result = apply_memref_layout(source, dict(inputs=[record, record], results=[mutation, view]))
    assert result.returncode != 0
    assert "cannot prove returned view fits" in result.stderr


def test_result_alias_cannot_bypass_borrowed_capacity_proof():
    source = """module {
      func.func @forward(%arg: memref<2x3xf32>, %value: memref<?x3xf32>) -> (memref<2x3xf32>, memref<?x3xf32>) {
        return %arg, %value : memref<2x3xf32>, memref<?x3xf32>
      }
    }"""
    record = dict(shape=[2, 3], stride=[3, 1], storage_offset=0)
    view = dict(record, shape=[-1, 3], alias_results=[0])
    result = apply_memref_layout(source, dict(inputs=[record, record], results=[dict(record, alias_inputs=[0]), view]))
    assert result.returncode != 0
    assert "cannot prove returned view fits" in result.stderr
