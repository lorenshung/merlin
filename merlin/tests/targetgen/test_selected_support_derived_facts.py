"""An explicitly selected support provider that ships no facts pin is served by the standard derivation
for its target -- a cache whose extraction committed to THIS provider's contract bytes, else a fresh
extraction -- and never by a legacy cache that cannot say whose contract it was extracted under."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from merlin.targetgen.rtl import circt_introspect, facts


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def selected(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(tmp_path / "repo"))
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.delenv("MERLIN_RTL_FACTS", raising=False)
    monkeypatch.delenv("MERLIN_TARGET_CONTRACT", raising=False)
    provider = tmp_path / "support"
    (provider / "contracts").mkdir(parents=True)
    (provider / "provider.yaml").write_text(
        "schema: merlin.provider.v1\nid: selected\ntarget: synthetic\nrole: support\n"
        "contract: contracts/target_contract.yaml\n"
    )
    contract = provider / "contracts" / "target_contract.yaml"
    contract.write_text("name: synthetic\n")
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(provider))
    monkeypatch.setattr(facts, "_written_by_another_family", lambda *args: False)
    facts.clear_resolution_cache()
    yield contract
    facts.clear_resolution_cache()


def _cache(*, contract: Path | None) -> Path:
    inputs = {"target": "synthetic"}
    if contract is not None:
        inputs.update(
            extractor_sha256=_sha(Path(circt_introspect.__file__).resolve()),
            extraction_contract_sha256=_sha(contract),
        )
    doc = {
        "generator": {"name": "merlin.targetgen.rtl.circt_introspect"},
        "inputs": inputs,
        "facts": {"target": "synthetic", "datapaths": [{"name": "input", "dtype": "i8"}]},
    }
    path = facts.rtl_facts_path("synthetic")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return path


def test_a_cache_extracted_under_the_selected_contract_serves_it(selected, monkeypatch):
    monkeypatch.setattr(facts, "_dump_facts_for_kind", lambda *a: pytest.fail("re-extracted a valid cache"))
    path = _cache(contract=selected)
    assert facts.find_facts("synthetic") == path
    assert facts.load_facts("synthetic")["facts"]["datapaths"]


def test_a_legacy_or_other_contract_cache_is_never_served_and_is_re_derived(selected, monkeypatch):
    _cache(contract=None)
    assert facts.find_facts("synthetic") is None  # discovery never extracts and never borrows
    derived = []

    def extract(path, target):
        derived.append(target)
        _cache(contract=selected)

    monkeypatch.setattr(facts, "_dump_facts_for_kind", extract)
    assert facts.ensure_facts("synthetic") == facts.rtl_facts_path("synthetic") and derived == ["synthetic"]
    # A cache extracted under another contract's bytes is not this provider's either.
    selected.write_text("name: synthetic\nversion: 2\n")
    facts.clear_resolution_cache()
    assert facts.find_facts("synthetic") is None


def test_a_derivation_that_cannot_run_is_unavailable_evidence_with_its_cause(selected, monkeypatch):
    def extract(path, target):
        raise RuntimeError("no elaboration on this host")

    monkeypatch.setattr(facts, "_dump_facts_for_kind", extract)
    with pytest.raises(FileNotFoundError, match="deriving them failed: RuntimeError: no elaboration"):
        facts.ensure_facts("synthetic")
