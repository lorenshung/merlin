"""A mode ledger cannot shrink the selected decoder population or certify itself."""

from __future__ import annotations

import copy
import json

import yaml

from merlin.targetgen.cli import main as targetgen_main
from merlin.targetgen.isa_mode_audit import audit_mode_inventory


def _inputs():
    controls = ["N"] * 17
    census = {
        "schema": "merlin.isa_source_census.v1",
        "rtl_revision": "a" * 40,
        "source_revision_verification": {
            "status": "verified",
            "rtl_revision": "a" * 40,
            "model_revision": "b" * 40,
        },
        "summary": {
            "patterns_not_decoded": [],
            "decoder_rows_without_pattern": [],
            "model_classes_without_compatible_pattern": [],
            "overlapping_patterns": [],
            "dma_kind_conflicts": [],
        },
        "rows": [
            {"name": "MOVE_A", "pattern_bits": "0" * 32, "decode_controls": controls},
            {"name": "MOVE_B", "pattern_bits": "1" * 32, "decode_controls": controls},
        ],
    }
    inventory = {
        "selected_sources": {"rtl_revision": "a" * 40, "model_revision": "b" * 40},
        "parameter_domains": {"registers": "selected target register domain"},
        "variants": [
            {
                "id": name,
                "rtl_bitpat": bit * 32,
                "rtl_decode_controls": ",".join(controls),
                "required": True,
                "dialect_op": "demo.move",
                "mode_attrs": {"kind": kind},
                "parameter_domains": ["registers"],
                "software_admitted": True,
                "blocked": [],
            }
            for name, bit, kind in (("MOVE_A", "0", "a"), ("MOVE_B", "1", "b"))
        ],
    }
    plan = {
        "target": "demo",
        "dialect_name": "demo",
        "types": [],
        "lowering": [],
        "ops": [
            {
                "name": "move",
                "signature": {
                    "operands": [{"name": "src", "type": "i32"}],
                    "results": [{"name": "dst", "type": "i32"}],
                    "attributes": [
                        {"name": "kind", "type": "string", "role": "mode", "choices": ["a", "b"]},
                        {"name": "register_index", "type": "i32", "role": "binding", "min": 0, "max": 7},
                    ],
                    "effects": [],
                },
            }
        ],
    }
    return census, inventory, plan


def test_parameterized_operation_accounts_for_every_decoder_mode():
    census, inventory, plan = _inputs()
    report = audit_mode_inventory(census, inventory, dialect_plan=plan)
    assert report["mode_inventory_ready"]
    assert report["typed_mode_binding_ready"]
    assert report["counts"]["typed_mode_bindings"] == 2
    assert report["counts"]["required_modes"] == 2
    assert report["counts"]["selected_decoder_modes"] == 2
    assert report["qualification"].startswith("source/mode/typed-plan accounting")


def test_mode_binding_checks_typed_attribute_domain_and_role():
    census, inventory, plan = _inputs()
    inventory["variants"][0]["mode_attrs"]["kind"] = "other"
    report = audit_mode_inventory(census, inventory, dialect_plan=plan)
    assert not report["mode_inventory_ready"]
    assert report["counts"]["typed_binding_problem_kinds"] == {"mode_attribute_out_of_domain": 1}

    inventory["variants"][0]["mode_attrs"] = {}
    report = audit_mode_inventory(census, inventory, dialect_plan=plan)
    assert report["counts"]["typed_binding_problem_kinds"] == {"required_mode_attribute_missing": 1}

    inventory["variants"][0]["mode_attrs"] = {"kind": "a", "register_index": 2}
    report = audit_mode_inventory(census, inventory, dialect_plan=plan)
    assert report["counts"]["typed_binding_problem_kinds"] == {"mode_attribute_not_in_plan": 1}

    plan["ops"][0]["signature"]["attributes"][0]["role"] = "binding"
    inventory["variants"][0]["mode_attrs"] = {"kind": "a"}
    report = audit_mode_inventory(census, inventory, dialect_plan=plan)
    assert report["counts"]["typed_binding_problem_kinds"] == {"mode_attribute_not_in_plan": 2}


def test_name_only_plan_cannot_complete_machine_mode_binding():
    census, inventory, plan = _inputs()
    plan["ops"][0].pop("signature")
    report = audit_mode_inventory(census, inventory, dialect_plan=plan)
    assert report["source_reconciled"]
    assert not report["typed_mode_binding_ready"]
    assert not report["mode_inventory_ready"]
    assert report["typed_plan_error"] == "dialect plan has no reviewed typed signatures"


def test_revision_string_without_verified_selected_bytes_is_not_source_bound():
    census, inventory, plan = _inputs()
    census["source_revision_verification"] = {"status": "unverified"}
    report = audit_mode_inventory(census, inventory, dialect_plan=plan)
    assert not report["source_bound"]
    assert report["source_discrepancies"] == ["selected_source_revisions_unverified"]
    census["source_revision_verification"] = {
        "status": "verified",
        "rtl_revision": "a" * 40,
        "model_revision": "c" * 40,
    }
    report = audit_mode_inventory(census, inventory, dialect_plan=plan)
    assert report["source_discrepancies"] == ["selected_source_revision_disagrees"]


def test_missing_changed_and_duplicate_modes_do_not_shrink_denominator():
    census, inventory, plan = _inputs()
    inventory["variants"].pop()
    report = audit_mode_inventory(census, inventory, dialect_plan=plan)
    assert not report["source_bound"]
    assert report["counts"]["selected_decoder_modes"] == 2
    assert report["counts"]["required_modes"] == 2
    assert report["modes"][1]["problems"] == ["selected_mode_missing_from_inventory"]

    census, inventory, plan = _inputs()
    inventory["variants"][0]["rtl_bitpat"] = "1" * 32
    inventory["variants"][1]["rtl_decode_controls"] = ",".join(["Y"] + ["N"] * 16)
    inventory["variants"].append(copy.deepcopy(inventory["variants"][1]))
    report = audit_mode_inventory(census, inventory, dialect_plan=plan)
    assert not report["source_bound"]
    assert report["counts"]["problem_kinds"] == {
        "duplicate_mode_id": 1,
        "rtl_decode_controls_changed": 1,
        "rtl_pattern_changed": 1,
    }


def test_admission_plan_and_parameter_unknowns_remain_visible():
    census, inventory, plan = _inputs()
    inventory["variants"][0]["software_admitted"] = False
    inventory["variants"][0]["blocked"] = ["numerical_semantics"]
    inventory["variants"][1]["parameter_domains"] = ["unknown"]
    inventory["variants"][1]["dialect_op"] = "demo.missing"
    report = audit_mode_inventory(census, inventory, dialect_plan=plan)
    assert report["source_bound"]
    assert not report["mode_inventory_ready"]
    assert report["counts"]["problem_kinds"] == {
        "dialect_operation_not_in_plan": 1,
        "mode_qualification_open": 1,
        "parameter_domain_unresolved": 1,
        "software_admission_missing": 1,
    }


def test_model_disagreement_is_not_hidden_by_exact_rtl_mode_match():
    census, inventory, plan = _inputs()
    census["summary"]["model_classes_without_compatible_pattern"] = ["MODEL_MODE"]
    inventory["variants"][0]["model_classes"] = ["MODEL_MODE"]
    census["rows"][0]["model_candidates"] = []
    report = audit_mode_inventory(census, inventory, dialect_plan=plan)
    assert report["source_bound"]
    assert not report["source_reconciled"]
    assert not report["mode_inventory_ready"]
    assert report["census_discrepancies"] == {"model_classes_without_compatible_pattern": ["MODEL_MODE"]}
    assert report["counts"]["problem_kinds"] == {"model_encoding_disagrees": 1}


def test_selected_rtl_resolution_requires_exact_reviewed_discrepancy():
    census, inventory, plan = _inputs()
    census["summary"]["model_classes_without_compatible_pattern"] = ["MODEL_MODE"]
    inventory["source_resolutions"] = [
        {
            "kind": "model_classes_without_compatible_pattern",
            "item": "MODEL_MODE",
            "authority": "selected_rtl",
            "reviewed": True,
            "evidence": "selected-core/mode-test",
        }
    ]
    report = audit_mode_inventory(census, inventory, dialect_plan=plan)
    assert report["source_reconciled"]
    assert report["mode_inventory_ready"]
    assert report["unresolved_source_discrepancies"] == []
    inventory["source_resolutions"][0]["item"] = "OLD_MODE"
    import pytest

    with pytest.raises(ValueError, match="stale, invented, or duplicated"):
        audit_mode_inventory(census, inventory, dialect_plan=plan)
    inventory["source_resolutions"][0]["item"] = "MODEL_MODE"
    inventory["source_resolutions"][0]["reviewed"] = False
    with pytest.raises(ValueError, match="reviewed selected_rtl"):
        audit_mode_inventory(census, inventory, dialect_plan=plan)


def test_cli_refuses_open_modes_and_clears_stale_output_on_malformed_input(tmp_path, capsys):
    census, inventory, plan = _inputs()
    census_file = tmp_path / "census.json"
    inventory_file = tmp_path / "inventory.json"
    plan_file = tmp_path / "plan.yaml"
    output = tmp_path / "audit.json"
    census_file.write_text(json.dumps(census))
    inventory_file.write_text(json.dumps(inventory))
    plan_file.write_text(yaml.safe_dump(plan))
    args = [
        "audit-dialect-modes",
        "--census",
        str(census_file),
        "--inventory",
        str(inventory_file),
        "--dialect-plan",
        str(plan_file),
        "--out",
        str(output),
    ]
    assert targetgen_main(args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "MODE_INVENTORY_READY"
    first_inventory_digest = json.loads(output.read_text())["inputs_sha256"]["inventory"]
    inventory["variants"][0]["software_admitted"] = False
    inventory_file.write_text(json.dumps(inventory))
    assert targetgen_main(args) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "MODE_OBLIGATIONS_OPEN"
    assert output.is_file()
    assert json.loads(output.read_text())["inputs_sha256"]["inventory"] != first_inventory_digest
    inventory_file.write_text("invalid json")
    assert targetgen_main(args) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "FAIL"
    assert not output.exists()
