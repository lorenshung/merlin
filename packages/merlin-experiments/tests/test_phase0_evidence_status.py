"""Evidence is `verified` only with nothing on record against it (operator policy, 2026-10-01)."""

from __future__ import annotations

import hashlib
import json

from merlin_experiments.phase0 import evidence_status as ES

_FACTS = b'{"facts": {"arrays": []}}\n'
_HARDWARE = b"target: fixture\n"


def _audit(**override):
    report = {
        "schema": ES.AUDIT_SCHEMA,
        "status": "verified",
        "facts_sha256": hashlib.sha256(_FACTS).hexdigest(),
        "hardware_spec_sha256": hashlib.sha256(_HARDWARE).hexdigest(),
        "checks": [{"source_status": "verified", "extraction": {"status": "agrees"}, "gap": None}],
        **override,
    }
    return json.dumps(report).encode()


def test_an_audit_must_verify_exactly_the_selected_bytes():
    assert ES.rtl_audit_diagnostic(_audit(), facts_raw=_FACTS, hardware_raw=_HARDWARE) is None
    refusals = [
        ES.rtl_audit_diagnostic(None, facts_raw=_FACTS, hardware_raw=_HARDWARE),
        ES.rtl_audit_diagnostic(_audit(), facts_raw=_FACTS + b" ", hardware_raw=_HARDWARE),
        ES.rtl_audit_diagnostic(_audit(), facts_raw=_FACTS, hardware_raw=None),
        ES.rtl_audit_diagnostic(_audit(status="unknown"), facts_raw=_FACTS, hardware_raw=_HARDWARE),
        ES.rtl_audit_diagnostic(_audit(schema="other"), facts_raw=_FACTS, hardware_raw=_HARDWARE),
        ES.rtl_audit_diagnostic(
            _audit(checks=[{"source_status": "verified", "extraction": {"status": "disagrees"}, "gap": None}]),
            facts_raw=_FACTS,
            hardware_raw=_HARDWARE,
        ),
        ES.rtl_audit_diagnostic(b"not json", facts_raw=_FACTS, hardware_raw=_HARDWARE),
    ]
    assert all(row is not None and row["component"] == "rtl-audit" for row in refusals)


def test_status_is_verified_only_without_diagnostics():
    assert ES.status([]) == "verified"
    assert ES.status([{"component": "facts", "status": "unknown", "reason": "absent"}]) == "diagnostic"


def test_an_unattested_capture_is_a_diagnostic():
    applications = {"app": {"capture_sha256": "a" * 64, "capture_receipt": {"receipt_sha256": "b" * 64}}}
    rows = ES.capture_attestation_diagnostics(applications, {})
    assert [row["component"] for row in rows] == ["capture-execution"] and rows[0]["application"] == "app"
    forged = {"app": {"schema": "merlin.capture_execution_attestation.v1", "status": "verified_sealed_execution"}}
    assert ES.capture_attestation_diagnostics(applications, forged)[0]["component"] == "capture-execution"
