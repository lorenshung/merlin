"""Exact exterior initialization with complete interior overwrite and live tensors."""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.insert_slice_destination import exterior_slabs, rewrite_module
from merlin.xdsl_dialects._common import text


@pytest.mark.parametrize(
    "shape,offsets,sizes",
    [
        ((7,), (2,), (3,)),
        ((4, 5), (1, 1), (2, 3)),
        ((2, 5, 7), (0, 1, 2), (2, 3, 4)),
        ((1, 6, 7, 3), (0, 1, 2, 0), (1, 4, 3, 3)),
        ((2, 3), (0, 0), (2, 3)),
        ((2, 3), (0, 0), (1, 1)),
    ],
)
def test_complement_is_disjoint_and_complete(shape, offsets, sizes):
    visits = np.zeros(shape, dtype=np.int32)
    for start, extent in exterior_slabs(shape, offsets, sizes):
        visits[tuple(slice(o, o + n) for o, n in zip(start, extent))] += 1
    interior = tuple(slice(o, o + n) for o, n in zip(offsets, sizes))
    assert np.all(visits[interior] == 0)
    visits[interior] += 1
    assert np.all(visits == 1)


@pytest.mark.parametrize(
    "shape,offsets,sizes", [((2,), (-1,), (1,)), ((2,), (1,), (2,)), ((2,), (0,), (0,)), ((2,), (0, 0), (1,))]
)
def test_invalid_regions_rejected(shape, offsets, sizes):
    with pytest.raises(ValueError):
        exterior_slabs(shape, offsets, sizes)


@pytest.mark.parametrize("border", [False, True])
@pytest.mark.parametrize("live_pad", [False, True])
def test_native_full_output_and_live_pad(tmp_path, border, live_pad):
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.pipeline import lower_to_llvm_ir
    from merlin.llvmlower.toolchain import clang

    source = Path(__file__).with_name("insert_destination_fixture.mlir").read_text()
    source = source.replace("@padded", "@forward").replace(
        "-> tensor<4x5xi8> {", "-> tensor<4x5xi8> attributes {llvm.emit_c_interface} {"
    )
    source = source.replace("arith.constant 0 : i8", "arith.constant -7 : i8")
    if live_pad:
        source = source.replace("-> tensor<4x5xi8> attributes", "-> (tensor<4x5xi8>,tensor<4x5xi8>) attributes")
        source = source.replace(
            "return %result : tensor<4x5xi8>", "return %result,%pad : tensor<4x5xi8>,tensor<4x5xi8>"
        )
    module = parse_mlir_text(source)
    assert len(rewrite_module(module, initialize_border_only=border)) == 1
    llvm = lower_to_llvm_ir(
        text(module, generic=True),
        workdir=tmp_path / "lower",
        vectorize=False,
        features=frozenset({"reuse_tensor_destination"}),
    )
    ll = tmp_path / "model.ll"
    ll.write_text(llvm)
    lib = tmp_path / f"model_{border}_{live_pad}.so"
    subprocess.run(
        [str(clang()), "-O2", "-fPIC", "-shared", str(ll), str(mlir_runtime_c()), "-o", str(lib)], check=True
    )
    inp = np.array([[-126, -1, 0], [1, 5, 126]], dtype=np.float32)
    expected = np.full((4, 5), -7, dtype=np.int8)
    expected[1:3, 1:4] = inp.astype(np.int8)
    invoke = HostModel.load(str(lib))
    for _ in range(10):
        out = np.full((4, 5), 99, dtype=np.int8)
        old = np.full((4, 5), 99, dtype=np.int8)
        arrays = [inp, out, old] if live_pad else [inp, out]
        invoke([(a.ctypes.data, a.shape) for a in arrays])
        np.testing.assert_array_equal(out, expected)
        if live_pad:
            np.testing.assert_array_equal(old, np.full((4, 5), -7, dtype=np.int8))


def test_border_option_preserves_scalar_body_and_refuses_live_producer():
    source = Path(__file__).with_name("insert_destination_fixture.mlir").read_text()
    module = parse_mlir_text(source)
    before = next(op for op in module.walk() if op.name == "linalg.generic")
    rewrite_module(module, initialize_border_only=True)
    after = next(op for op in module.walk() if op.name == "linalg.generic")
    assert [op.name for op in before.body.block.ops] == [op.name for op in after.body.block.ops]
    # The original scalar producer is erased, and its exact body is cloned.
    assert text(before.body.block.first_op) == text(after.body.block.first_op)
    live = source.replace("-> tensor<4x5xi8> {", "-> (tensor<4x5xi8>,tensor<2x3xi8>) {").replace(
        "return %result : tensor<4x5xi8>", "return %result,%q : tensor<4x5xi8>,tensor<2x3xi8>"
    )
    module = parse_mlir_text(live)
    old = text(module)
    assert rewrite_module(module, initialize_border_only=True) == []
    assert text(module) == old


def test_normal_feature_requires_destination_alias_hygiene_only_when_selected():
    from merlin.llvmlower.impr_features import normalize
    from merlin.llvmlower.selfcopy import FEATURE

    assert normalize(frozenset()) == frozenset()
    selected = normalize({"initialize_tensor_border_only"})
    assert {"initialize_tensor_border_only", "reuse_tensor_destination", FEATURE} <= selected


@pytest.mark.parametrize("blocking", [False, True])
def test_normal_preparation_selects_border_option(tmp_path, blocking):
    import json

    from merlin.llvmlower.impr_features import normalize
    from merlin.runtime.backends.zephyr_model import prepare_for_lowering

    source = tmp_path / "source.mlir"
    source.write_text(Path(__file__).with_name("insert_destination_fixture.mlir").read_text())
    (tmp_path / "build").mkdir()
    prepared, features = prepare_for_lowering(
        source, tmp_path / "build", features=normalize({"initialize_tensor_border_only"}), blocking=blocking
    )
    receipt = json.loads((tmp_path / "build/insert_slice_destination/receipt.json").read_text())
    assert len(receipt["routes"]) == 1
    assert receipt["routes"][0]["storage"] == "fresh exterior slabs with fully overwritten interior"
    assert "reuse_tensor_destination" in features
    parse_mlir_text(prepared.read_text()).verify()
