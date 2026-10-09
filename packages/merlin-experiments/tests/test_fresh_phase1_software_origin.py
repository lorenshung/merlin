"""Fresh authoring refuses legacy software metadata before runtime admission.

These controls use an actual stock FIRRTL producer and its replayed structural
intake. They neither author a compiler nor substitute an experimental runtime.
"""

from __future__ import annotations

import json
import os
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase0.rtl_intake import (
    RtlIntakePin,
    RtlIntakeRefusal,
    issue_independent_hardware_intake,
)
from merlin_experiments.phase0.software_intake import IndependentSoftwareIntake, issue_independent_software_intake
from merlin_experiments.phase1.component_origin import FreshPhase1Inputs
from merlin_experiments.phase2.contracts import StageGateError

from merlin.targetgen.rtl import source_selection
from merlin.targetgen.target_experiment import load_target_experiment


@pytest.fixture
def hardware(tmp_path):
    selected = os.environ.get("MERLIN_TEST_FIRTOOL")
    if not selected:
        pytest.skip("requires explicitly selected stock FIRRTL producer")
    firrtl = tmp_path / "unit.fir"
    firrtl.write_text(
        "FIRRTL version 3.2.0\ncircuit Unit :\n"
        "  module Unit : @[generators/private_control/src/Unit.scala 1:1]\n"
        "    input clock : Clock\n"
        "    smem bank : UInt<8>[4] @[generators/private_control/src/Unit.scala 3:1]\n"
    )
    descriptor = tmp_path / "target.yaml"
    descriptor.write_text("target: private_control\n")
    bundle = source_selection.produce_selection(
        target="private_control",
        firrtl=firrtl,
        generator="private_control",
        config="TestConfiguration",
        core_root="Unit",
        firtool=Path(selected).resolve(strict=True),
        output=tmp_path / "production",
    )
    protected = tmp_path / "protected"
    protected.mkdir()
    return issue_independent_hardware_intake(
        target="private_control",
        descriptor=descriptor,
        source_bundle=bundle,
        forbidden_roots=(protected,),
        output=tmp_path / "hardware",
    )


def inputs(hardware, software):
    members = {field.name: None for field in fields(FreshPhase1Inputs)}
    return FreshPhase1Inputs(**{**members, "hardware": hardware, "software": software})


@pytest.mark.parametrize("claim", ["missing", "reviewed_hash", "matching_object"])
def test_software_status_and_matching_hash_cannot_admit_fresh_author_inputs(hardware, claim):
    software = {
        "missing": None,
        "reviewed_hash": {"status": "reviewed", "sha256": "a" * 64, "hardware_sha256": hardware.sha256},
        "matching_object": SimpleNamespace(hardware=hardware, sha256="a" * 64),
    }[claim]
    with pytest.raises(StageGateError, match="independently reviewed minimal software authority"):
        inputs(hardware, software).verify()


def test_reconstructing_typed_software_record_cannot_mint_live_origin(hardware, tmp_path):
    software = IndependentSoftwareIntake(
        hardware,
        RtlIntakePin("caller-source", str(tmp_path / "source.json"), "a" * 64),
        (),
        b"{}",
        b'{"status":"reviewed"}',
    )
    with pytest.raises(RtlIntakeRefusal, match="live independently issued"):
        inputs(hardware, software).verify()
    foreign = replace(software, hardware=SimpleNamespace(sha256=hardware.sha256))
    with pytest.raises(StageGateError, match="matching independently"):
        inputs(hardware, foreign).verify()


def test_actual_protected_software_selection_preserves_missing_runtime_gate(tmp_path):
    selected = os.environ.get("MERLIN_TEST_MINIMAL_SOFTWARE_SELECTION")
    if not selected:
        pytest.skip("requires explicit independently reviewed minimal source selection")
    configuration = json.loads(Path(selected).read_text())
    protected = tmp_path / "protected"
    protected.mkdir()
    hardware = issue_independent_hardware_intake(
        **configuration["hardware"], forbidden_roots=(protected,), output=tmp_path / "hardware"
    )
    software = issue_independent_software_intake(
        hardware=hardware, **configuration["software"], forbidden_roots=(protected,), output_root=tmp_path / "software"
    )
    target = load_target_experiment(configuration["hardware"]["descriptor"], source_root=tmp_path)
    admitted_selection = replace(inputs(hardware, software), target_experiment=target)
    assert software.hardware is hardware
    # Actual independently selected semantics cannot substitute for a complete
    # target runtime or authorize authoring while its effects remain UNKNOWN.
    with pytest.raises(StageGateError, match="independently issued target runtime support"):
        admitted_selection.verify()
    with pytest.raises(RtlIntakeRefusal, match="live independently issued"):
        replace(admitted_selection, software=replace(software)).verify()
