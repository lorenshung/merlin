"""Completed source consumption rechecks capture issuers without executing them."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from merlin_experiments.corpus import preparation
from merlin_experiments.phase0 import evidence as evidence_module
from merlin_experiments.spec import SpecError


def _selected_evidence(applications: dict, attestations: dict) -> tuple[dict, SimpleNamespace]:
    views = json.dumps(
        {
            "application_inventory": {"applications": applications},
            "capture_execution_attestations": attestations,
        },
        sort_keys=True,
    ).encode()
    observed = SimpleNamespace(status="verified", raw_facts_sha256=None, views_json=views, diagnostics=[])
    plan = {
        "phase0_evidence_bundle": "/historical/phase0",
        "phases": {
            "0": {
                "phase0_evidence": {
                    "status": "verified",
                    "raw_facts_sha256": None,
                    "views_sha256": hashlib.sha256(views).hexdigest(),
                    "diagnostics": [],
                }
            }
        },
    }
    return plan, observed


def test_completed_source_rejects_changed_declared_capture_issuer(tmp_path, monkeypatch):
    applications = {"selected": {"capture_sha256": "0" * 64, "capture_receipt": {"receipt_sha256": "1" * 64}}}
    plan, observed = _selected_evidence(applications, {"selected": {"status": "verified_sealed_execution"}})
    monkeypatch.setattr(preparation, "generation_lineage", lambda *_: {"capsules": 1})
    monkeypatch.setattr(evidence_module, "load_exported_evidence", lambda _: observed)

    with pytest.raises(SpecError, match="capture execution changed"):
        preparation._verify_completed_generation(plan, tmp_path)


def test_completed_source_rejects_changed_generation_time_capture_issuer(tmp_path, monkeypatch):
    plan, observed = _selected_evidence({}, {})
    member = tmp_path / "model" / "selected"
    member.mkdir(parents=True)
    (member / "capsule.yaml").write_text(
        yaml.safe_dump({"name": "selected", "source": "pytorch", "capture_execution_attestations": [{}]})
    )
    monkeypatch.setattr(preparation, "generation_lineage", lambda *_: {"capsules": 1})
    monkeypatch.setattr(evidence_module, "load_exported_evidence", lambda _: observed)

    with pytest.raises(SpecError, match="generation-time capture attestation changed"):
        preparation._verify_completed_generation(plan, tmp_path)
