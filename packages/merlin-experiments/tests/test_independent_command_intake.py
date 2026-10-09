"""Actual source tracking and native command observation issuance without adapters."""

import dataclasses
import json
import shutil
import subprocess

import pytest
from merlin_experiments.phase0.command_intake import IndependentCommandIntake, issue_independent_command_intake
from merlin_experiments.phase0.rtl_intake import RtlIntakeRefusal, issue_independent_hardware_intake
from test_independent_rtl_intake import selected  # noqa: F401 -- shared actual replay fixture


@pytest.fixture
def command_selection(selected, tmp_path):  # noqa: F811 -- registered shared pytest fixture
    hardware = issue_independent_hardware_intake(**selected)
    checkout = tmp_path / "hardware-source"
    checkout.mkdir()
    source = checkout / "ISA.scala"
    source.write_text(
        "object ISA {\n  // codes\n  val MOVE = 17.U\n  val EXECUTE = 3.U\n  val END = 0.U\n"
        "  class Packed extends Bundle {\n    val spare = UInt(7.W)\n    val address = UInt(5.W)\n  }\n}\n"
    )
    for args in (
        ["init", "-q"],
        ["config", "user.name", "Fixture"],
        ["config", "user.email", "fixture@example.invalid"],
        ["remote", "add", "origin", "https://example.invalid/public-hardware"],
        ["add", "ISA.scala"],
        ["commit", "-qm", "public fixture source"],
    ):
        subprocess.run(["git", "-C", str(checkout), *args], check=True, capture_output=True)
    commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"]).decode().strip()
    # This native fixture serializes the same empty Unit as the replay fixture;
    # it tests actual calls/provenance, not CIRCT correctness or target behavior.
    generic = """builtin.module {
      "hw.module"() ({ "hw.output"() : () -> () })
      {sym_name = "Unit", module_type = !hw.modty<>} : () -> ()
    }"""
    c = tmp_path / "serialize.c"
    c.write_text(
        "#include <stdio.h>\n#include <string.h>\nint main(int n, char **v) {\n"
        'for(int i=1;i+1<n;i++) if(!strcmp(v[i],"-o")) { FILE *f=fopen(v[i+1],"w");\n'
        "if(!f) return 2; fputs(" + json.dumps(generic) + ",f); return fclose(f)!=0; } return 1; }\n"
    )
    tool = tmp_path / "serializer"
    subprocess.run([shutil.which("cc"), str(c), "-o", str(tool)], check=True, capture_output=True)
    return {
        "hardware": hardware,
        "checkout": checkout,
        "commit": commit,
        "isa_source": source,
        "function_span": ("// codes", "END"),
        "circt_opt": tool,
        "forbidden_roots": selected["forbidden_roots"],
        "output": tmp_path / "command-intake",
    }


def test_actual_tracked_source_and_native_generic_observation(command_selection):
    intake = issue_independent_command_intake(**command_selection)
    facts = intake.public_facts()
    assert facts["hardware_intake_sha256"] == command_selection["hardware"].sha256
    assert facts["source_declarations"]["values"] == [3, 17]
    assert facts["source_bundle_layouts"]["Packed"]["width"] == 12
    assert facts["hw_observations"]["complete_isa"] is False
    assert "instruction_effects_and_numerical_semantics" in facts["unknowns"]
    intake.verify_public_facts(command_selection["output"] / "facts.json")


def test_constructor_and_changed_issued_object_cannot_grant_authority(command_selection):
    intake = issue_independent_command_intake(**command_selection)
    copy = IndependentCommandIntake(
        intake.hardware, intake.source_pins, intake.git_json, intake.facts_json, intake.receipt_json
    )
    with pytest.raises(RtlIntakeRefusal, match="live independently issued"):
        copy.verify()
    with pytest.raises(RtlIntakeRefusal, match="live independently issued"):
        dataclasses.replace(intake, facts_json=b"{}").verify()
    object.__setattr__(intake, "git_json", b"{}")
    with pytest.raises(RtlIntakeRefusal, match="live independently issued"):
        intake.verify()


def test_tracked_modification_and_wrong_commit_refuse(command_selection):
    source = command_selection["isa_source"]
    source.write_text(source.read_text() + "// changed\n")
    with pytest.raises(RtlIntakeRefusal, match="tracked modifications"):
        issue_independent_command_intake(**command_selection)
    source.write_text(source.read_text().removesuffix("// changed\n"))
    command_selection["commit"] = "0" * 40
    with pytest.raises(RtlIntakeRefusal, match="exact checkout"):
        issue_independent_command_intake(**command_selection)


def test_public_extra_fields_and_source_drift_refuse(command_selection):
    intake = issue_independent_command_intake(**command_selection)
    public = command_selection["output"] / "extra-facts.json"
    facts = intake.public_facts()
    facts["injected_schedule"] = {"tile_size": 12}
    public.write_text(json.dumps(facts))
    with pytest.raises(RtlIntakeRefusal, match="complete issued projection"):
        intake.verify_public_facts(public)
    source = command_selection["isa_source"]
    source.write_text(source.read_text() + "// source drift\n")
    with pytest.raises(RtlIntakeRefusal, match="source changed"):
        intake.verify()


def test_protected_source_refuses_before_tracked_source_commands(command_selection, monkeypatch):
    from merlin_experiments.phase0 import command_intake

    private = command_selection["forbidden_roots"][0] / "ISA.scala"
    private.write_text("private sentinel")
    command_selection["isa_source"] = private
    monkeypatch.setattr(command_intake, "_tracked_source", lambda *a: pytest.fail("private source examined"))
    with pytest.raises(RtlIntakeRefusal, match="protected implementation"):
        issue_independent_command_intake(**command_selection)
