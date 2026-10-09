"""``merlin-targetgen quant-contract`` reads only a saved evidence bundle's inventoried bytes.

The contract an adapter receives is named by digest in its policy, so the same evidence must give
the same bytes on every run, and a bundle member changed since export must be refused rather than
projected into a contract.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from merlin.targetgen import cli
from merlin.targetgen import quant_adapter_contract as QAC
from merlin.targetgen.quantization_spec import build_quantization_contract
from merlin.targetgen.software_spec import validate_software_spec


def _spec():
    return validate_software_spec(
        {
            "schema": "merlin.software_spec.v1",
            "target": "synthetic",
            "numerical_semantics": {
                "model": {"engine": "specir_fp_reduce"},
                "operand_dtype": "fp8_e4m3",
                "accumulator_dtype": "bf16",
                "readout_dtype": "bf16",
                "product_rounding": "accumulator_format",
                "rounding": "rne",
                "reduction_order": "index_sequential",
                "reduction_cadence": "per_step",
                "subnormal_operand_flush": True,
                "internal_arithmetic": {"operand_domain": {"exponent_range": [1, 15], "reserved_codes": [127, 255]}},
            },
            "operations": {"contraction": {"placement": "accelerator", "operand_dtypes": ["fp8_e4m3"]}},
            "quantization": {
                "formats": [
                    {
                        "operand_dtype": "fp8_e4m3",
                        "accumulator_dtype": "bf16",
                        "eligible_operations": ["contraction"],
                        "scale_encoding": "unknown",
                    }
                ]
            },
        }
    )


def _bundle(root):
    spec = _spec()
    members = {
        "software/software-spec.json": json.dumps(spec, sort_keys=True).encode(),
        "software/quantization-contract.json": json.dumps(
            build_quantization_contract(spec, {"quantization_candidates": []}), sort_keys=True
        ).encode(),
    }
    for name, raw in members.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(raw)
    inventory = {
        name: {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)} for name, raw in members.items()
    }
    manifest = {"schema": "phase0_evidence_v1", "target": "synthetic", "artifacts": inventory}
    (root / "evidence-manifest.json").write_text(json.dumps(manifest))
    return root


def _run(capsys, *argv):
    assert cli.main(["quant-contract", *argv]) == 0
    return json.loads(capsys.readouterr().out)


def test_the_same_evidence_writes_the_same_contract_bytes(tmp_path, capsys):
    bundle = _bundle(tmp_path / "evidence")
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    report = _run(capsys, "--evidence-bundle", str(bundle), "--format-id", "fp8_e4m3__bf16", "--out", str(first))
    _run(capsys, "--evidence-bundle", str(bundle), "--format-id", "fp8_e4m3__bf16", "--out", str(second))
    assert first.read_bytes() == second.read_bytes()
    assert report["sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    contract = json.loads(first.read_bytes())
    assert contract == QAC.from_evidence_bundle(bundle, format_id="fp8_e4m3__bf16")
    assert contract["operand"]["finite_max"] == 448.0
    assert contract["status"] == report["status"] == "diagnostic"
    assert "scale.encoding" in report["unknowns"]


def test_default_output_is_a_versioned_product_under_the_declared_concern(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    bundle = _bundle(tmp_path / "evidence")
    report = _run(capsys, "--evidence-bundle", str(bundle), "--format-id", "fp8_e4m3__bf16")
    from pathlib import Path

    concern = tmp_path / "out" / "artifacts" / "handoff" / "quantization-contracts" / "synthetic" / "v1"
    path = Path(report["path"])
    assert path.is_relative_to(concern) and path.parent.parent == concern
    assert path.name == "adapter-contract.json" and path.is_file()
    assert (path.parent / "manifest.yaml").is_file()


def test_a_member_changed_since_export_is_refused(tmp_path):
    bundle = _bundle(tmp_path / "evidence")
    member = bundle / "software/quantization-contract.json"
    member.write_bytes(member.read_bytes().replace(b'"unknown"', b'"e8m0"', 1))
    with pytest.raises(ValueError, match="changed since export"):
        QAC.from_evidence_bundle(bundle, format_id="fp8_e4m3__bf16")
