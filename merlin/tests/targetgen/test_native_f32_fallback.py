"""Native word admission is structural and independent of sample bytes."""

import numpy as np
import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen.core_aten_components import prepare_integer_components


@pytest.mark.parametrize("inactive", [False, True])
@pytest.mark.parametrize("word_map", ["negate", "literal_clamp"])
def test_native_f32_word_admission_never_reads_sample_or_eager(tmp_path, monkeypatch, word_map, inactive):
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
                "%literal = arith.constant 0.0 : f32\n"
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

    records = [record]
    unrelated = dict(status="not_in_shard", input_abi=[dict(dtype="bf16")])
    if inactive:
        records = [unrelated, record, dict(status="not_selected")]
    decision = prepare_integer_components(
        tmp_path,
        dict(cases=records),
        dict(spec={}, host={}),
        Provider(),
        None,
        module,
        limb_attribute="test.limb",
    )
    assert unrelated == dict(status="not_in_shard", input_abi=[dict(dtype="bf16")])
    assert decision is None
    # A literal compare/select clamp is a declared floating activation: it is
    # refused here before any provider schedule request (the quantized byte
    # activation path owns it). A negation asks the provider for a schedule.
    assert bool(called) == (word_map == "negate")
    first_called = list(called)
    assert not (tmp_path / "source-ieee-closure.json").exists()
    # No admissibility or generated-kernel property may use these observations.
    (tmp_path / "inputs.npz").write_bytes(b"not a readable tensor archive")
    record["native_eager_boundary"] = {"poison": "never read"}
    called.clear()
    prepare_integer_components(
        tmp_path,
        dict(cases=records),
        dict(spec={}, host={}),
        Provider(),
        None,
        parse_mlir_text(source),
        limb_attribute="test.limb",
    )
    assert called == first_called
    assert (tmp_path / "model.mlir").read_text() == source
