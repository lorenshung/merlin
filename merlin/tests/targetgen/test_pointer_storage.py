"""Explicit original logical storage declarations, without shaped allocation."""

import base64
from dataclasses import replace

import pytest

from merlin.runtime.direct_kernel_harness import DirectKernelAbi, render_direct_kernel
from merlin.targetgen.contract.compile_only import CompileOnlySourceAbi, CompileOnlyTensor
from merlin.targetgen.contract.pointer_storage import OriginalPointerStorageContract, PointerStoragePolicy
from merlin.targetgen.contract.readback_policy import FULL_VALUES_B64, ReadbackPolicy


def policy():
    return PointerStoragePolicy(
        "row_major_contiguous", "inputs_then_outputs", "disjoint", "static_original_tensor_type", "little", 8
    )


def original(shape=(2, 3), dtype="i32"):
    return CompileOnlySourceAbi(
        (CompileOnlyTensor("input", shape, dtype),), (CompileOnlyTensor("output", shape, dtype),)
    )


def buffer():
    return {
        "tensors": {name: {"dtype": "i32", "shape": [2, 3]} for name in ("input", "output")},
        "kernel_abi": {
            "kind": "whole_program",
            "outputs": ["output"],
            "args": [{"tensor": "input", "access": "read"}, {"tensor": "output", "access": "write"}],
        },
    }


def test_complete_original_large_extents_and_order_derive_without_data():
    contract = OriginalPointerStorageContract(original((1 << 50, 7)), policy())
    assert contract.slots[0].element_count == (1 << 50) * 7
    assert contract.slots[0].byte_extent == (1 << 50) * 7 * 4
    assert contract.slots[0].element_strides == (7, 1)
    assert [(slot.role, slot.name) for slot in contract.slots] == [("input", "input"), ("output", "output")]
    assert contract.record()["schema"] == "merlin.original_pointer_storage.v1"


@pytest.mark.parametrize(
    "field,value",
    [
        ("layout", "candidate_layout"),
        ("argument_order", "outputs_first"),
        ("slot_aliasing", "may_alias"),
        ("extent_derivation", "candidate_metadata"),
        ("byte_order", "native"),
        ("tensor_alignment", 0),
        ("tensor_alignment", 3),
        ("tensor_alignment", True),
    ],
)
def test_no_layout_order_alias_width_or_alignment_defaults(field, value):
    with pytest.raises(ValueError, match="explicit supported software ABI choice"):
        replace(policy(), **{field: value}).record()


@pytest.mark.parametrize("shape,dtype", [((0, 3), "i32"), ((2, 3), "f32"), ((2, 3), "i4"), ((), "i32")])
def test_unknown_packed_float_empty_storage_is_not_invented(shape, dtype):
    with pytest.raises(ValueError):
        OriginalPointerStorageContract(original(shape, dtype), policy()).record()


def test_candidate_pointer_order_cannot_select_original_storage():
    selected = OriginalPointerStorageContract(original(), policy())
    cb = buffer()
    assert selected.bind_candidate(cb)["pointer_arity"] == 2
    cb["kernel_abi"]["args"].reverse()
    with pytest.raises(ValueError, match="pointer order/access"):
        selected.bind_candidate(cb)


def test_real_shared_pointer_renderer_enforces_selected_original_abi():
    selected = OriginalPointerStorageContract(original(), policy())
    cb = buffer()
    cb["tensors"]["input"]["preload_b64"] = base64.b64encode(bytes(24)).decode()
    kwargs = dict(
        inputs={"input": {}},
        readback_policy=ReadbackPolicy(FULL_VALUES_B64),
        abi=DirectKernelAbi("entry", None, 8, "little", "void"),
    )
    text = render_direct_kernel(cb, original_storage=selected, **kwargs)
    assert "tensor_0[24]" in text and "tensor_1[24]" in text
    with pytest.raises(ValueError, match="byte order/alignment"):
        render_direct_kernel(
            cb, original_storage=replace(selected, policy=replace(policy(), byte_order="big")), **kwargs
        )
    with pytest.raises(ValueError, match="exact independent typed declaration"):
        render_direct_kernel(cb, original_storage=selected.record(), **kwargs)
