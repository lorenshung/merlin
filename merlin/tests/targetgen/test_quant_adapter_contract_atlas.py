"""The example target's authored spec projects to a diagnostic adapter contract.

Its operand domain states exponent fields 1..14 but no reserved codes, operand rounding or scale
encoding, so the finite maximum an adapter would use is not established by the spec and the
contract must say so instead of choosing a value.
"""

from __future__ import annotations

import hashlib

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen import quant_adapter_contract as QAC
from merlin.targetgen.quantization_spec import build_quantization_contract
from merlin.targetgen.software_spec import load_software_spec

pytestmark = pytest.mark.target("atlas")


def test_todays_spec_projects_to_a_diagnostic_contract():
    path = repo_root() / "examples/atlas/target/software-spec.yaml"
    spec = load_software_spec(path, target="atlas")
    quantization = build_quantization_contract(spec, {"quantization_candidates": [], "readout_facets": []})
    source = {"path": str(path.relative_to(repo_root())), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    contract = QAC.build(spec, quantization, format_id=quantization["formats"][0]["id"], spec_source=source)
    assert contract["status"] == "diagnostic"
    assert contract["operand"]["exponent_range"] == [1, 14]
    assert contract["operand"]["subnormal_flush"] is True
    assert contract["operand"]["finite_max"] == "unknown"
    for field in ("operand.reserved_codes", "operand.finite_max", "scale.encoding", "scale.block_size"):
        assert field in contract["unknowns"]
    assert [row["operation"] for row in contract["families"]["accelerator"]] == ["contraction"]
