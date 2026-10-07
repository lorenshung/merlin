"""Explicit post-offload callbacks preserve routing identity and default behavior."""

import json
from types import SimpleNamespace

import pytest

from merlin.llvmlower.device_build import apply_post_offload_transform
from merlin.llvmlower.device_shim import emit_dense_translation_unit


def test_absent_callback_needs_no_files_and_changes_nothing(tmp_path):
    source = tmp_path / "absent.mlir"
    assert apply_post_offload_transform(None, source, tmp_path / "absent.json", tmp_path / "out") == source
    assert not (tmp_path / "out").exists()


def test_callback_gets_exact_routing_and_records_selected_bytes(tmp_path):
    source = tmp_path / "source.mlir"
    source.write_text("original")
    sidecar = tmp_path / "routing.json"
    sidecar.write_text('{"routed":1}')

    def callback(actual, identity, work):
        assert actual == source and identity == sidecar
        result = work / "selected.mlir"
        result.write_text("selected")
        return result

    output = apply_post_offload_transform(
        SimpleNamespace(post_offload_transform=callback), source, sidecar, tmp_path / "out"
    )
    receipt = json.loads((tmp_path / "out/post_offload_transform.json").read_text())
    assert output.read_text() == "selected" and source.read_text() == "original"
    assert receipt["source_sha256"] != receipt["selected_sha256"]


@pytest.mark.parametrize("failure", ["missing", "identity"])
def test_failed_callback_cannot_silently_change_routing(tmp_path, failure):
    source = tmp_path / "source.mlir"
    source.write_text("original")
    sidecar = tmp_path / "routing.json"
    sidecar.write_text("{}")

    def callback(actual, identity, work):
        if failure == "identity":
            identity.write_text('{"changed":true}')
        return work / "absent"

    with pytest.raises(ValueError):
        apply_post_offload_transform(
            SimpleNamespace(post_offload_transform=callback), source, sidecar, tmp_path / "out"
        )
    assert not (tmp_path / "out/post_offload_transform.json").exists()


def test_dense_emitter_identity_needs_explicit_kernel_full_write_fact():
    args = ("fixture", {"entry": (2, 3, 4)}, {"entry": ("i8", "i8", "i32")})
    plain = emit_dense_translation_unit(*args, kernel_symbol_for=lambda _: "kernel")
    qualified = emit_dense_translation_unit(
        *args, kernel_symbol_for=lambda _: "kernel", kernel_fully_written_arguments=(2,)
    )
    assert not plain.writer_contracts and plain.text == qualified.text
    assert qualified.writer_contracts[0]["result_argument"] == 2
    assert qualified.writer_contracts[0]["fully_written_arguments"] == [2]
    with pytest.raises(ValueError):
        emit_dense_translation_unit(*args, kernel_symbol_for=lambda _: "kernel", kernel_fully_written_arguments=(0, 2))
