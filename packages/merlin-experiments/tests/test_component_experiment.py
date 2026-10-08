"""Mutation checks for minimal views, isolation refusal and final comparison arithmetic."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import component_experiment as C
from merlin_experiments.phase2.contracts import StageGateError

from merlin.targetgen.compiler_library import freeze_compiler_library


def _view(tmp_path):
    root = tmp_path / "installed"
    (root / "merlin").mkdir(parents=True)
    (root / "merlin/__init__.py").write_text("")
    (root / "merlin/portable.py").write_text("def lower(value):\n    return value\n")
    (root / "private-answer.txt").write_text("MUST_NOT_COPY")
    library = freeze_compiler_library(
        root,
        review_id="review",
        public_modules=("merlin.portable",),
        sources=(("merlin/__init__.py", "merlin"), ("merlin/portable.py", "merlin.portable")),
    )
    source = tmp_path / "generated.mlir"
    source.write_text("module {}\n")
    member = C.ApprovedInput(
        source, "generated_input/case.mlir", hashlib.sha256(source.read_bytes()).hexdigest(), "generated_input"
    )
    return C.materialize_component_view(
        tmp_path / "agent-view", library=library, library_root=root, inputs=(member,), generation_sha256="1" * 64
    )


def test_view_copies_only_explicit_reviewed_members_without_source_paths(tmp_path):
    view = _view(tmp_path)
    record = C.verify_component_view(view)
    assert len(record["members"]) == 3
    payload = (view.root / "manifest.json").read_text()
    assert str(tmp_path) not in payload
    assert not (view.root / "private-answer.txt").exists()


@pytest.mark.parametrize("mutation", ["history", "bytes", "link", "empty-directory"])
def test_added_history_or_changed_member_invalidates_view(tmp_path, mutation):
    view = _view(tmp_path)
    source = view.root / "generated_input/case.mlir"
    if mutation == "history":
        (view.root / ".git").mkdir()
        (view.root / ".git/config").write_text("remote private-repository")
    elif mutation == "bytes":
        source.chmod(0o644)
        source.write_text("changed")
    elif mutation == "link":
        source.unlink()
        source.symlink_to(tmp_path / "generated.mlir")
    else:
        (view.root / "unreviewed").mkdir()
    with pytest.raises(StageGateError):
        C.verify_component_view(view)


def test_strict_policy_does_not_inherit_network_home_or_entire_checkout(tmp_path):
    view = _view(tmp_path)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    tool = tmp_path / "tool"
    tool.write_bytes(b"synthetic-runtime")
    grant = C.RuntimeGrant(tool, "/usr/bin/tool", hashlib.sha256(tool.read_bytes()).hexdigest())
    argv = C.strict_tool_policy(view, candidate, runtime=(grant,))
    assert "--unshare-all" in argv and "--clearenv" in argv
    assert "--share-net" not in argv
    assert str(tmp_path / "installed") not in argv
    assert not any(value in argv for value in ("/home", "/scratch", ".git"))
    with pytest.raises(StageGateError, match="overlaps"):
        C.strict_tool_policy(view, view.root, runtime=(grant,))


def test_unavailable_namespace_probe_is_not_pass(monkeypatch):
    monkeypatch.setattr(
        C.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1, stdout=b"", stderr=b"not permitted")
    )
    with pytest.raises(StageGateError, match="unavailable"):
        C.run_isolation_probe(("bwrap", "--unshare-all"), ("/usr/bin/true",))


def _comparisons():
    return tuple(
        C.FinalMemberComparison(name, 1000, 1050, "1" * 64, "2" * 64, "3" * 64, True, True, True)
        for name in ("heldout-a", "heldout-b", "heldout-c")
    )


def _gate(rows=None, **kw):
    values = {
        "expected_members": ("heldout-a", "heldout-b", "heldout-c"),
        "phase12_wall_s": 10,
        "handwritten_wall_s": 200,
    }
    values.update(kw)
    return C.final_component_campaign_gate(_comparisons() if rows is None else rows, **values)


def test_final_parity_is_per_member_and_uses_exact_boundary_arithmetic():
    assert _gate()["status"] == "pass"
    rows = list(_comparisons())
    rows[0] = replace(rows[0], candidate_cycles=1051)
    rows[1] = replace(rows[1], candidate_cycles=1)
    assert _gate(tuple(rows))["status"] == "fail"
    assert _gate(phase12_wall_s=10.01)["status"] == "fail"


def test_unknown_hardware_or_historical_time_cannot_pass():
    rows = list(_comparisons())
    rows[0] = replace(rows[0], hardware_verified=False)
    assert _gate(tuple(rows))["status"] == "unknown"
    assert _gate(handwritten_wall_s=None)["status"] == "unknown"
    rows[0] = replace(rows[0], accuracy_passed=False)
    assert _gate(tuple(rows))["status"] == "fail"


def test_final_membership_and_numeric_evidence_are_not_coerced():
    with pytest.raises(StageGateError, match="membership"):
        _gate(_comparisons()[:2])
    rows = list(_comparisons())
    rows[0] = replace(rows[0], candidate_cycles=True)
    with pytest.raises(StageGateError, match="integers"):
        _gate(tuple(rows))
    with pytest.raises(StageGateError, match="finite"):
        _gate(phase12_wall_s=float("nan"))
