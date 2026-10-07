"""Explicit normal preparation, public ABI and immutable source boundaries."""

import json
from pathlib import Path

import numpy as np
import pytest

from merlin.common.digest import sha256_file
from merlin.llvmlower.prepared_model_transform import RECEIPT, apply_prepared_model_transform

SOURCE = """module {
  func.func @forward(%a: tensor<5xf32>) -> tensor<5xf32> {
    %e = tensor.empty() : tensor<5xf32>
    %r = linalg.generic {
      indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>],
      iterator_types = ["parallel"]
    } ins(%a : tensor<5xf32>) outs(%e : tensor<5xf32>) {
    ^bb0(%x: f32, %unused: f32):
      %c = arith.constant 2.0 : f32
      %y = arith.addf %x, %c : f32
      linalg.yield %y : f32
    } -> tensor<5xf32>
    return %r : tensor<5xf32>
  }
}"""


def _source(tmp_path):
    source = tmp_path / "source.mlir"
    source.write_text(SOURCE)
    return source


def test_empty_selection_does_not_read_or_create_work(tmp_path):
    source = tmp_path / "absent.mlir"
    work = tmp_path / "absent-work"
    assert apply_prepared_model_transform(source, work, None) == source
    assert not work.exists()


def test_callback_receives_private_immutable_snapshot_and_retains_public_abi(tmp_path):
    source = _source(tmp_path)
    work = tmp_path / "selection"
    seen = []

    def selected(snapshot, output):
        assert snapshot != source and snapshot.read_bytes() == source.read_bytes()
        seen.append(snapshot)
        path = output / "selected.mlir"
        path.write_text(snapshot.read_text().replace("2.0", "3.0"))
        return path

    result = apply_prepared_model_transform(source, work, selected)
    assert len(seen) == 1 and source.read_text() == SOURCE
    assert "3.0" in result.read_text()
    receipt = json.loads((work / RECEIPT).read_text())
    assert receipt["source_sha256"] == sha256_file(source)
    assert receipt["selected_sha256"] == sha256_file(result)
    assert receipt["public_definitions"] == ["forward"]


@pytest.mark.parametrize("mutation", ["snapshot", "original", "public_abi", "invalid", "outside", "symlink"])
def test_invalid_selection_does_not_publish_success(tmp_path, mutation):
    source = _source(tmp_path)
    work = tmp_path / "selection"

    def selected(snapshot, output):
        path = output / "selected.mlir"
        path.write_text(SOURCE)
        if mutation == "snapshot":
            snapshot.write_text(SOURCE + "\n")
        elif mutation == "original":
            source.write_text(SOURCE + "\n")
        elif mutation == "public_abi":
            path.write_text(SOURCE.replace("tensor<5xf32>", "tensor<6xf32>"))
        elif mutation == "invalid":
            path.write_text("module { func.func @broken() { return %unknown : f32 } }")
        elif mutation == "outside":
            return source
        elif mutation == "symlink":
            path.unlink()
            path.symlink_to(source)
        return path

    with pytest.raises(Exception):
        apply_prepared_model_transform(source, work, selected)
    assert not (work / RECEIPT).exists()


def test_normal_baremetal_build_selects_prepared_mlir_and_records_it(tmp_path):
    from merlin.llvmlower import toolchain
    from merlin.runtime.backends import spike, spike_model

    if not spike.available() or not toolchain.m2m_python().is_file() or not toolchain.clang().is_file():
        pytest.skip("bare-metal and upstream toolchain unavailable")
    bundle, work = tmp_path / "bundle", tmp_path / "build"
    bundle.mkdir()
    (bundle / "model.mlir").write_text(SOURCE)
    (bundle / "weights.safetensors.manifest.json").write_text('{"0":{"kind":"input","name":"values"}}')
    values = np.array([-5, -0.75, 0, 1.25, 7], dtype=np.float32)
    np.savez(bundle / "inputs.npz", in0=values)
    seen = []

    def selected(source, directory):
        from xdsl.dialects.arith import ConstantOp
        from xdsl.dialects.builtin import FloatAttr, f32

        from merlin.frontends.linalg_mlir import parse_mlir_file
        from merlin.xdsl_dialects._common import text

        seen.append(source)
        # Deliberately different arithmetic establishes actual callback routing.
        # The generic seam checks ABI/identity, not numerical equivalence.
        path = directory / "selected.mlir"
        module = parse_mlir_file(source)
        changed = 0
        for op in module.walk():
            if isinstance(op, ConstantOp) and isinstance(op.value, FloatAttr) and op.value.value.data == 2:
                op.properties["value"] = FloatAttr(3, f32)
                changed += 1
        assert changed == 1
        path.write_text(text(module, generic=True))
        return path

    built = spike_model.build(
        bundle,
        work,
        arena_mb=1,
        prepared_model_transform=selected,
        cflags_override=["-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-builtin"],
    )
    assert len(seen) == 1 and (bundle / "model.mlir").read_text() == SOURCE
    record = json.loads((work / "compilation_recipe.json").read_text())
    assert record["status"] == "completed"
    bound = record["preparation"]["prepared_model_transform"]
    assert bound["sha256"] == sha256_file(Path(bound["path"]))
    result = spike_model.run(built["elf"], mem_bytes=built["mem_bytes"], isa="rv64gc", timeout=60)
    np.testing.assert_array_equal(result["outputs"], values + np.float32(3))
