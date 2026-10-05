"""The selected gemmini support provider's contracts are Merlin's reviewed reference contracts plus
the provider's own plugin block -- nothing else may diverge, and where they would, the reviewed one wins.

A provider that lagged the reference silently changed every build: without
``encoding.corpus_issue_order`` each package group's interface capsule failed and the whole model was
declined to the CPU; its readout scaling granularity and DMA bound disagreed with the reviewed facts."""

from __future__ import annotations

import pytest
import selected_driver
import yaml

from merlin.common.paths import repo_root

pytestmark = pytest.mark.target("gemmini")
_TARGET = "gemmini"


def _pair(name: str) -> tuple[dict, dict]:
    support = selected_driver.require_support(_TARGET)
    reviewed = yaml.safe_load((repo_root() / "examples" / _TARGET / "target" / "contracts" / name).read_text())
    provided = yaml.safe_load((support / "contracts" / name).read_text())
    return reviewed, provided


@pytest.mark.parametrize("name", ["target_contract.yaml", "residual.yaml"])
def test_the_provider_contract_is_the_reviewed_one_plus_its_plugin(name):
    reviewed, provided = _pair(name)
    assert provided.get("plugin"), "the provider owns the executable plugin declaration"
    assert {k: v for k, v in provided.items() if k != "plugin"} == {k: v for k, v in reviewed.items() if k != "plugin"}


def test_the_fields_a_build_depends_on_agree():
    reviewed, provided = _pair("target_contract.yaml")
    assert provided["encoding"]["corpus_issue_order"] == reviewed["encoding"]["corpus_issue_order"]
    dma = reviewed["memory_model"]["dma"]["max_transfer_bytes"]
    assert provided["memory_model"]["dma"]["max_transfer_bytes"] == dma

    def scalings(contract):
        units = contract["compute_units"]
        rows = units.values() if isinstance(units, dict) else units
        return [row.get("readout", row).get("scaling") for row in rows if isinstance(row, dict)]

    assert any(scalings(reviewed)) and scalings(provided) == scalings(reviewed)
