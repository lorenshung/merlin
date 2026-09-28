"""Selected-source extraction must not mistake missing or mixed evidence for qualification."""

from copy import deepcopy
from types import SimpleNamespace

from merlin.common.digest import sha256_file
from merlin.targetgen.rtl import circt_introspect as ci
from merlin.targetgen.rtl import facts as resolver
from merlin.targetgen.rtl import introspect, mlc_bridge


def test_declared_firrtl_drives_cell_reader_and_unresolved_formats_remain_unknown(tmp_path, monkeypatch):
    fir = tmp_path / "selected.fir"
    fir.write_text(
        "FIRRTL version 6.0.0\n"
        "circuit Grid :\n"
        "  module Cell :\n"
        "    input clock : Clock\n"
        "    output io : { flip a : UInt<8>, flip b : UInt<8>, flip addend : UInt<16>, mac : UInt<16>}\n"
    )
    monkeypatch.setattr(ci, "elaborated_firrtl", lambda target: [])
    monkeypatch.setattr(
        introspect, "declared_rtl_source", lambda target: SimpleNamespace(artifacts=lambda: {"fir": fir})
    )
    monkeypatch.setattr(mlc_bridge, "compute_unit_kinds", lambda target: ("systolic",))
    body = {"arrays": [{"name": "mesh", "element": "Cell"}]}
    assert ci._datapaths_from_cells("test-grid", body)
    assert [item["elem_bits"] for item in body["datapaths"]] == [8, 16]
    assert all(item["dtype"] is None for item in body["datapaths"])
    assert body["source"]["firrtl_inputs"] == [{"path": str(fir.resolve()), "sha256": sha256_file(fir)}]
    result = ci.validate({"facts": body}, {"compute_units": [{"name": "grid", "kind": "systolic"}]})
    assert not result["agree"]
    assert len(result["unknown"]) >= 2

    def invalid_declaration(target):
        raise introspect.RtlSourceInvalid("selected source is unavailable")

    monkeypatch.setattr(introspect, "declared_rtl_source", invalid_declaration)
    monkeypatch.setattr(ci, "elaborated_firrtl", lambda target: [fir])
    refused = {"arrays": [{"name": "mesh", "element": "Cell"}]}
    assert ci._datapaths_from_cells("test-grid", refused) == []
    assert "datapaths" not in refused
    assert any("selected source is unavailable" in item for item in refused["datapaths_undeterminable"])


def test_missing_and_absent_contract_evidence_cannot_agree():
    result = ci.validate({"facts": {}}, {"compute_units": [{"name": "grid", "kind": "systolic"}]})
    assert not result["agree"] and not result["diverge"]
    assert any("no RTL datapath" in item for item in result["unknown"])
    assert introspect.validate_against_contract({"datapaths": [{"name": "input", "dtype": "i8"}]}, {}) == [
        "UNKNOWN: contract declares no compute_units to cross-check"
    ]


def test_cli_validation_reports_unavailable_contract(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ci, "dump_facts", lambda *args, **kwargs: {"facts": {}})
    monkeypatch.setattr(resolver, "target_contract_path", lambda target: tmp_path / "absent-contract.yaml")
    assert ci.main(["--target", "test-grid", "--out", str(tmp_path / "facts.json"), "--validate"]) == 0
    assert "contract: selected contract is unavailable" in capsys.readouterr().out


def test_every_consumed_source_invalidates_cache_even_without_feature_gates(tmp_path):
    paths = {role: tmp_path / role for role in ("hw", "core_hw", "fir", "hierarchy", "isa")}
    for role, path in paths.items():
        path.write_text(role)
    inputs = {}
    for role, path in paths.items():
        inputs[f"{role}_path"] = str(path)
        inputs[f"{role}_sha256"] = sha256_file(path)
    document = {"inputs": inputs, "facts": {"arrays": [{"name": "mesh"}]}}
    assert resolver._declared_cache_pins_match(document)
    for role, path in paths.items():
        path.write_text("changed")
        assert not resolver._declared_cache_pins_match(document), role
        path.write_text(role)
    census_only = deepcopy(document)
    census_only["inputs"].pop("fir_path")
    census_only["inputs"].pop("fir_sha256")
    census_only["facts"]["source"] = {"fir_path": str(paths["fir"]), "fir_sha256": sha256_file(paths["fir"])}
    assert resolver._declared_cache_pins_match(census_only)
    paths["fir"].write_text("changed")
    assert not resolver._declared_cache_pins_match(census_only)
    paths["fir"].write_text("fir")
    document["inputs"]["reader_sources"] = [{"path": str(paths["isa"]), "sha256": sha256_file(paths["isa"])}]
    document["inputs"].pop("isa_path")
    document["inputs"].pop("isa_sha256")
    assert resolver._declared_cache_pins_match(document)
    paths["isa"].write_text("changed reader")
    assert not resolver._declared_cache_pins_match(document)
    assert resolver._declared_cache_pins_match({"inputs": {}, "facts": {"arrays": [{"name": "mesh"}]}})


def test_build_records_exact_sources_and_does_not_qualify_mixed_elaborations(tmp_path, monkeypatch):
    core, fir, hierarchy, isa = [tmp_path / name for name in ("core.mlir", "full.fir", "hierarchy.json", "isa.scala")]
    for path in (core, fir, hierarchy, isa):
        path.write_text(path.name)
    monkeypatch.setattr(resolver, "target_contract_path", lambda target: tmp_path / "absent-contract.yaml")
    monkeypatch.setattr(ci, "_accumulator_layout", lambda target: None)
    monkeypatch.setattr(introspect, "declared_rtl_source", lambda target: SimpleNamespace(root=tmp_path))
    monkeypatch.setattr(
        introspect,
        "sim_elaboration_facts",
        lambda target, root: (
            {"fir": fir},
            {"source": {"fir_path": str(fir), "hierarchy_path": str(hierarchy), "config": "selected"}},
        ),
    )
    monkeypatch.setattr(mlc_bridge, "core_hw_mlir", lambda target: core)
    for name in (
        "_facts_from_discovery",
        "_memories_from_declared_elaboration",
        "_memories_from_ports",
        "_datapaths_from_cells",
        "_timing_from_discovery",
    ):
        monkeypatch.setattr(ci, name, lambda target, body: [])
    monkeypatch.setattr(ci, "_funct_name_table", lambda *args: None)
    monkeypatch.setattr(ci, "extract_funct_table_via_decoder", lambda target: None)
    monkeypatch.setattr(ci, "extract_elaborated_rtl_features", lambda *args: {"status": "unknown"})
    rec = ci.build_facts(target="test-grid", hw_path=tmp_path / "absent.mlir", isa_path=isa)
    assert rec["inputs"]["core_hw_path"] == str(core.resolve())
    assert rec["inputs"]["fir_path"] == str(fir.resolve())
    assert rec["inputs"]["hierarchy_sha256"] == sha256_file(hierarchy)
    assert rec["source_consistency"]["status"] == "unverified"
    assert any("no shared production" in item for item in ci.validate(rec)["unknown"])
