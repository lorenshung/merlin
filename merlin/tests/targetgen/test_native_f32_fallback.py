"""Finite f32 data need not inherit overflow in a calibration observer."""

import json

import numpy as np
import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen.core_aten_components import prepare_integer_components


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("word_map", ["negate", "literal_clamp"])
def test_native_f32_is_explicit_and_keeps_the_independent_bit_gate(tmp_path, monkeypatch, native, word_map):
    monkeypatch.setattr(
        "merlin.targetgen.transfer_contracts.screen_transfer_contract", lambda *args, **kwargs: dict(status="admitted")
    )
    module = parse_mlir_text("""module {
      func.func @forward(%x: tensor<5xf32>) -> (tensor<5xf32>,tensor<5xf32>) {
        %e=tensor.empty() : tensor<5xf32>
        %v=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],
                          iterator_types=["parallel"]}
          ins(%x : tensor<5xf32>) outs(%e : tensor<5xf32>) {
          ^bb0(%a:f32,%unused:f32):
            %r=arith.negf %a : f32
            linalg.yield %r : f32
        } -> tensor<5xf32>
        return %v,%x : tensor<5xf32>,tensor<5xf32>
      }}""")
    if word_map == "literal_clamp":
        module = parse_mlir_text(
            str(module).replace(
                "%r = arith.negf %a : f32",
                "%zero = arith.constant 0 : i32\n"
                "%literal = arith.sitofp %zero : i32 to f32\n"
                "%less = arith.cmpf olt, %a, %literal : f32\n"
                "%r = arith.select %less, %literal, %a : f32",
            )
        )
    if word_map == "literal_clamp":
        assert any(op.name == "arith.select" for op in module.walk())
    source = str(module) + "\n"
    (tmp_path / "model.mlir").write_text(source)
    x = np.array(
        [np.finfo(np.float32).max, -np.finfo(np.float32).max, np.nextafter(np.float32(0), np.float32(1)), 0.0, -0.0],
        np.float32,
    )
    np.savez(tmp_path / "inputs.npz", in0=x)
    expected = -x if word_map == "negate" else np.where(x < 0, np.float32(0), x)
    record = dict(
        status="bundled",
        input_abi=[dict(dtype="f32")],
        output_indices=[0],
        semantic_boundary=dict(post_indices=[1]),
        native_eager_boundary=dict(
            authority="untransformed_native_cpu_eager",
            output_bytes=[expected.tobytes().hex()],
            post_input_bytes=[x.tobytes().hex()],
        ),
    )
    called = []

    class Provider:
        def admit_numeric_schedule(self, source, **kwargs):
            called.append(source)
            assert "test.limb" in source
            return dict(status="declined", reason="test deliberately does not grant a device proof")

    decision = prepare_integer_components(
        tmp_path,
        dict(cases=[record]),
        dict(spec={}, host={}),
        Provider(),
        None,
        module,
        limb_attribute="test.limb",
        native_f32=native,
    )
    assert decision is None
    assert bool(called) == native
    if native:
        closure = json.loads((tmp_path / "source-ieee-closure.json").read_text())
        assert closure["output_bytes"] == [expected.tobytes().hex(), x.tobytes().hex()]
    assert (tmp_path / "model.mlir").read_text() == source
