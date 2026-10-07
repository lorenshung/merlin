"""Capture policy readers refuse ambiguity before replay or admission."""

import hashlib
import json

import pytest
from merlin_experiments.capture_execution import sealed_m2m, sealed_python, sealed_static
from merlin_experiments.phase0 import capture_selection


@pytest.mark.parametrize(
    "module,member",
    [
        (sealed_static, "capture_execution_attestation.json"),
        (sealed_python, "sealed_python_diagnostic.json"),
        (sealed_m2m, "sealed_m2m_pending.json"),
    ],
)
@pytest.mark.parametrize("raw", [b'{"issuer":"first","issuer":"second"}', b'{"ignored":NaN}', b'{"ignored":1e999}'])
def test_receipt_refuses_ambiguous_or_nonfinite_json_before_replay(tmp_path, module, member, raw):
    run = tmp_path / "run"
    run.mkdir()
    (run / member).write_bytes(raw)
    with pytest.raises(ValueError, match="unreadable"):
        module.replay_verify(run)


@pytest.mark.parametrize("extra", [',"extra":false,"extra":true', ',"extra":NaN', ',"extra":1e999'])
def test_preselected_hash_does_not_license_ambiguous_json(tmp_path, extra):
    root = tmp_path / "selection"
    root.mkdir(mode=0o700)
    plan = {"schema": sealed_m2m.SCHEMA}
    document = {
        "schema": capture_selection.SCHEMA,
        "status": "preselected_before_capture",
        "phase0_admission": "not_granted",
        "plan": plan,
        "plan_sha256": hashlib.sha256(capture_selection._json(plan)).hexdigest(),
        "checkpoint": {"kind": "none"},
    }
    raw = json.dumps(document).encode()
    path = root / capture_selection.MEMBER
    path.write_bytes(raw)
    path.chmod(0o400)
    assert capture_selection.load(path, expected_sha256=hashlib.sha256(raw).hexdigest()) == document
    changed = raw[:-1] + extra.encode() + b"}"
    path.chmod(0o600)
    path.write_bytes(changed)
    path.chmod(0o400)
    with pytest.raises(ValueError, match="unreadable"):
        capture_selection.load(path, expected_sha256=hashlib.sha256(changed).hexdigest())
