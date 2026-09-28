"""Saved source programs are reusable offline, byte-bound, and not target certificates."""

import hashlib
import json

import numpy as np
import pytest
import yaml
from merlin_experiments.corpus.coverage import selected_cohort_coverage
from merlin_experiments.phase0 import coverage_commitment as CC
from merlin_experiments.phase0 import writer
from merlin_experiments.phase0.evidence import _materialize_evidence
from merlin_experiments.phase0.requirements import _materialized_iteration_capsules, _validate_capture_recipes
from merlin_experiments.phase0.writer import _integer_reference_bound, _write_capsule

from merlin.targetgen import capsule_source as source
from merlin.targetgen.capsule_common import load_capsule
from merlin.targetgen.corpus_spec import CorpusBinding, build


def _bundle(root):
    root.mkdir()
    weights = root / "weights.safetensors"
    weights.write_bytes(b"diagnostic-fixture-weights")
    (root / "weights.safetensors.manifest.json").write_text(
        json.dumps(
            {
                "0": {"kind": "param", "name": "weight"},
                "1": {"kind": "input", "name": "x"},
            }
        )
    )
    program = (
        f'builtin.module attributes {{prov.weights_file = "{weights}"}} {{ '
        "func.func @forward(%weight: tensor<2xf32>, %x: tensor<2xf32>) -> tensor<2xf32> { "
        "func.return %x : tensor<2xf32> } }"
    )
    (root / "model.mlir").write_text(program)
    raw = program.encode()
    trace = {"mlir": {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}}
    (root / "frontend-trace.json").write_text(json.dumps(trace))
    meta = {
        "ok": True,
        "opaque": 0,
        "dtype": "fp32",
        "torch_seed": 0,
        "weights": str(weights),
        "input_abi": [{"shape": [2], "dtype": "f32"}],
        "output_abi": [{"shape": [2], "dtype": "f32"}],
        "frontend_trace": {
            "path": "frontend-trace.json",
            "sha256": hashlib.sha256((root / "frontend-trace.json").read_bytes()).hexdigest(),
        },
    }
    (root / "meta.json").write_text(json.dumps(meta))
    np.savez(root / "inputs.npz", in0=np.array([1, 2], dtype=np.float32))
    np.save(root / "golden.npy", np.array([1, 2], dtype=np.float32))
    (root / "input_order.json").write_text('{"x": 0}')
    receipt = {
        "schema": "m2m.capture-receipt.v1",
        "materialized_abi": {"complete": True},
        "source_closure_verified": False,
        "source": {"path": "/unavailable/loader.py"},
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in root.iterdir()
        },
    }
    (root / "capture_receipt.json").write_text(json.dumps(receipt))
    return {
        "capture_source_path": str(root / "model.mlir"),
        "capture_sha256": hashlib.sha256(raw).hexdigest(),
        "capture_receipt": {"receipt_sha256": hashlib.sha256((root / "capture_receipt.json").read_bytes()).hexdigest()},
        "n_operations": 3,
    }


def test_source_capsule_reuse_is_offline_exact_and_fail_closed(tmp_path, monkeypatch):
    application = _bundle(tmp_path / "original")
    entries, outputs = _materialized_iteration_capsules({"applications": {"iteration": application}}, "0" * 64)
    saved = tmp_path / "installed"
    _materialize_evidence(saved, outputs)
    selected = entries[0]
    selected["materialized_capture"]["path"] = str(saved / selected["materialized_capture"]["path"])
    # Deliberately remove the original source; only the immutable selected members may be read.
    for path in (tmp_path / "original").iterdir():
        path.unlink()
    monkeypatch.setattr(source, "PytorchRefSource", lambda *a, **kw: pytest.fail("unexpected recapture"))
    binding = CorpusBinding("fixture", 2, "f32", "f32", False, ["L0"], "tolerance_float", atol=1e-5, rtol=1e-5)
    result = _write_capsule(selected, binding, tmp_path / "corpus")
    capsule = yaml.safe_load((result / "capsule.yaml").read_text())
    assert load_capsule(result)["source_role"] == "model_derived"
    assert capsule["materialized_capture"]["source_closure_verified"] is False
    assert capsule["materialized_capture"]["coverage_scope"] == "full_capture"
    assert not (result / "capsule.pytorch.py").exists()
    assert capsule["expected"]["instruction_classes"] == []
    receipt = json.loads((result / "frontend-evidence.json").read_text())
    assert receipt["packaging"] == "weights_reference_relocation"
    assert receipt["source_mlir_sha256"] == application["capture_sha256"]
    relocation = receipt["weights_reference_relocation"]
    assert "from" not in relocation
    assert (
        relocation["from_reference_sha256"]
        == hashlib.sha256(str(tmp_path / "original/weights.safetensors").encode("utf-8")).hexdigest()
    )
    assert relocation["to"] == "capsule.weights.safetensors"
    assert str(tmp_path / "original/weights.safetensors") not in (result / "capsule.yaml").read_text()
    assert str(tmp_path / "original/weights.safetensors") not in (result / "frontend-evidence.json").read_text()
    assert 'prov.weights_file = "capsule.weights.safetensors"' in (result / "capsule.interface.mlir").read_text()
    assert yaml.safe_load((result / "golden.yaml").read_text())["outputs"] == {"Y0": [1.0, 2.0]}
    with pytest.raises(source.M2MUnavailable, match="held-out validation"):
        source.materialized_model_artifacts({**selected["materialized_capture"], "workload_role": "validation"})
    source_receipt = saved / "materialized/iteration/capture_receipt.json"
    original_receipt = source_receipt.read_bytes()
    renamed_receipt = json.loads(original_receipt)
    renamed_receipt["source"]["path"] = "/missing/workloads/tiny_llama/loader.py"
    source_receipt.write_text(json.dumps(renamed_receipt))
    with pytest.raises(source.M2MUnavailable, match="renamed"):
        source.materialized_model_artifacts(
            {
                **selected["materialized_capture"],
                "receipt_sha256": hashlib.sha256(source_receipt.read_bytes()).hexdigest(),
            }
        )
    source_receipt.write_bytes(original_receipt)
    selected_model = saved / "materialized/iteration/model.mlir"
    selected_model.write_text(selected_model.read_text() + "\n")
    with pytest.raises(source.M2MUnavailable, match="receipt"):
        source.materialized_model_artifacts(selected["materialized_capture"])


def test_derived_micro_model_reads_only_explicit_frozen_captures(tmp_path, monkeypatch):
    from merlin.targetgen import micro_model

    saved = tmp_path / "frozen" / "model.mlir"
    saved.parent.mkdir()
    saved.write_text("module {}\n")
    monkeypatch.setattr(writer, "_roster_captures", lambda: pytest.fail("ambient recaptures were read"))

    def spec(target, captures):
        assert target == "fixture"
        assert captures == {"iteration": saved}
        return micro_model.MicroModelSpec(target="fixture")

    monkeypatch.setattr(micro_model, "spec", spec)
    monkeypatch.setattr(micro_model, "emit_pytorch", lambda spec: "# frozen inventory\n")
    entry = {"cat": "model", "name": "SY_micro_model", "_frozen_application_captures": {"iteration": saved}}
    assert writer._emit_micro_model_loader(entry, "fixture", tmp_path / "out")
    assert entry["loader"].endswith("capsule.pytorch.py")
    assert "_frozen_application_captures" not in entry


def test_quantized_capture_recipe_must_match_selected_provider(tmp_path):
    bundle = tmp_path / "capture"
    _bundle(bundle)
    meta = json.loads((bundle / "meta.json").read_text())
    meta["quantization_stats"] = {"recipe_sha256": "a" * 64}
    (bundle / "meta.json").write_text(json.dumps(meta))
    captures = {"iteration": bundle / "model.mlir"}
    with pytest.raises(ValueError, match="iteration.*different quantization recipe"):
        _validate_capture_recipes(captures, {"b" * 64})
    _validate_capture_recipes(captures, {"a" * 64})


def test_integer_golden_bound_uses_concrete_reduction_and_internal_width():
    semantics = {
        "internal_arithmetic": {
            "full_operation_overflow_policy": "bounded_exact_requires_each_partial_sum",
            "mac_result_bits": 20,
            "signed_operand_bits": 8,
        }
    }
    capsule = {
        "operation": {"op": "matmul", "attributes": {"lhs": "A", "weight": "W"}},
        "inputs": [
            {"name": "A", "role": "input", "shape": [2, 4], "dtype": "i8"},
            {"name": "W", "role": "weight", "shape": [4, 2], "dtype": "i8"},
        ],
    }
    proof = _integer_reference_bound({"numerical_semantics": semantics}, capsule)
    assert proof["status"] == "proven_safe" and proof["reduction_extent"] == 4
    capsule["stimulus_range"] = [127, 127]
    capsule["inputs"][0]["shape"] = [2, 64]
    capsule["inputs"][1]["shape"] = [64, 2]
    with pytest.raises(ValueError, match="may_overflow"):
        _integer_reference_bound({"numerical_semantics": semantics}, capsule)


def test_exact_conformance_cohort_cannot_borrow_unselected_siblings(tmp_path):
    binding = CorpusBinding("fixture", 4, "int8", "i32", True, ["L0"], "exact_int")
    members = {}
    for name, size in (("selected", 4), ("unselected", 7)):
        entry = {
            "name": name,
            "cat": "isa",
            "kind": "isa",
            "op": "matmul",
            "label": "public",
            "M": size,
            "K": size,
            "N": size,
            "source_role": "derived_sweep",
            "source_reference": "fixture",
        }
        capsule, program = build(entry, binding)
        directory = tmp_path / "isa" / name
        directory.mkdir(parents=True)
        (directory / "capsule.yaml").write_text(yaml.safe_dump(capsule))
        (directory / "capsule.interface.mlir").write_text(program)
        members[name] = directory
    requirement = {"cells": [{"cell": "contraction/i8/partial"}], "boundaries": {"tile_edge": 4}}
    selected = selected_cohort_coverage(requirement, [members["selected"]])
    assert selected["n_covered"] == 0
    assert selected["uncovered"] == ["contraction/i8/partial"]
    broad = selected_cohort_coverage(requirement, [tmp_path / "isa"])
    assert broad["n_covered"] == 1
    inputs = {"schema": CC.INPUT_SCHEMA, "target": "fixture", "capability_contract": {}}
    assert (
        CC._selected_program({"kind": "layer", "interface_mlir": "capsule.interface.mlir"}, members["selected"])
        is not None
    )
    assert (
        CC._selected_program(
            {"kind": "layer", "interface_mlir": "capsule.interface.mlir", "linalg_mlir": "missing.mlir"},
            members["selected"],
        )
        is None
    )
    assert (
        CC._selected_program(
            {"kind": "layer", "interface_mlir": "capsule.interface.mlir", "linalg_mlir": None}, members["selected"]
        )
        is None
    )
    report = CC.observe_cohort(inputs, [members["selected"]], target="fixture")
    assert not [row for row in report["blockers"] if row["component"] == "capsule"]
    CC.verify_cohort_binding(report, inputs, [members["selected"]])
    interface = members["selected"] / "capsule.interface.mlir"
    interface.write_text(interface.read_text() + "\n")
    with pytest.raises(ValueError, match="exact admitted-cohort bytes"):
        CC.verify_cohort_binding(report, inputs, [members["selected"]])
    interface.write_text("malformed IR")
    malformed = CC.observe_cohort(inputs, [members["selected"]], target="fixture")
    assert any(row["reason"] == "capsule program is not inventoried" for row in malformed["blockers"])
