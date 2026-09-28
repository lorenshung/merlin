"""A real solver must prove a selected cell property and expose a false one."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from merlin_experiments.phase0.hardware_validation import prove_combinational_property, verify_property_receipt


def _tools() -> tuple[str, str]:
    circt = os.environ.get("MERLIN_TEST_CIRCT_OPT")
    z3 = os.environ.get("MERLIN_TEST_Z3")
    if not circt or not z3:
        pytest.skip("explicit CIRCT and Z3 tool selection required for hardware proof")
    return circt, z3


def test_selected_combinational_cell_is_proven_and_replayed(tmp_path: Path):
    circt, z3 = _tools()
    source = tmp_path / "selected.mlir"
    source.write_text(
        "module {\n"
        "  hw.module @Cell(in %a: i8, in %b: i8, out sum: i8) {\n"
        "    %sum = comb.add %a, %b : i8\n"
        "    hw.output %sum : i8\n"
        "  }\n}\n"
    )
    reference = tmp_path / "reference.mlir"
    reference.write_text(
        "hw.module @Reference(in %a: i8, in %b: i8, out sum: i8) {\n"
        "  %sum = comb.add %b, %a : i8\n"
        "  hw.output %sum : i8\n"
        "}\n"
    )
    good = prove_combinational_property(
        hw_source=source,
        module="Cell",
        reference=reference,
        reference_module="Reference",
        circt_opt=circt,
        z3=z3,
        output=tmp_path / "good",
        property_statement="commutative 8-bit modular addition for all inputs",
    )
    assert good["status"] == "proven", json.dumps(good)
    assert good["solver_verdict"] == "unsat"
    assert good["cycle_bound"] is None and good["assumptions"] == []
    assert good["whole_operation_support"] == "not_qualified"
    assert verify_property_receipt(tmp_path / "good/property.json", hw_source=source)["status"] == "proven"
    with pytest.raises(ValueError, match="assumptions require"):
        prove_combinational_property(
            hw_source=source,
            module="Cell",
            reference=reference,
            reference_module="Reference",
            circt_opt=circt,
            z3=z3,
            output=tmp_path / "refused",
            property_statement="assumption cannot be silently ignored",
            assumptions=["a is zero"],
        )
    source.write_text(source.read_text().replace("comb.add", "comb.mul"))
    with pytest.raises(ValueError, match="selected source digest"):
        verify_property_receipt(tmp_path / "good/property.json", hw_source=source)


def test_false_property_is_refuted_with_counterexample(tmp_path: Path):
    circt, z3 = _tools()
    source = tmp_path / "selected.mlir"
    source.write_text(
        "module {\n  hw.module @Cell(in %a: i8, in %b: i8, out sum: i8) {\n"
        "    %sum = comb.add %a, %b : i8\n    hw.output %sum : i8\n  }\n}\n"
    )
    reference = tmp_path / "reference.mlir"
    reference.write_text(
        "hw.module @Reference(in %a: i8, in %b: i8, out sum: i8) {\n"
        "  %sum = comb.mul %a, %b : i8\n  hw.output %sum : i8\n}\n"
    )
    bad = prove_combinational_property(
        hw_source=source,
        module="Cell",
        reference=reference,
        reference_module="Reference",
        circt_opt=circt,
        z3=z3,
        output=tmp_path / "bad",
        property_statement="false claim: addition equals multiplication",
    )
    assert bad["status"] == "refuted", json.dumps(bad)
    assert bad["solver_verdict"] == "sat"
    assert bad["counterexample"].startswith("sat\n(")
    with pytest.raises(ValueError, match="not formally proven"):
        verify_property_receipt(tmp_path / "bad/property.json", hw_source=source)
