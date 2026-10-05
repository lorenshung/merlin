"""A YAML document parsed once per its bytes is the same document every caller would have parsed."""

from __future__ import annotations

import pytest
import yaml

from merlin.common import yaml as Y


def test_a_parsed_document_is_shared_by_bytes_and_never_by_reference():
    text = "a: {b: [1, 2]}\n"
    first = Y.safe_load_text(text)
    first["a"]["b"].append(3)
    assert Y.safe_load_text(text) == {"a": {"b": [1, 2]}}, "a caller's mutation leaked into the next reader"
    assert Y.safe_load_text(text + "c: 1\n") == {"a": {"b": [1, 2]}, "c": 1}, "different bytes are a new parse"
    with pytest.raises(yaml.YAMLError):
        Y.safe_load_text("a: [")


def test_a_contract_read_through_the_registry_is_a_private_copy():
    from merlin.targetgen import target_registry as TR

    resolved = TR.resolve("gemmini")
    first = resolved.load_contract()
    first["__mutated__"] = True
    assert "__mutated__" not in resolved.load_contract()


def _tracked_yaml():
    import subprocess

    from merlin.common.paths import repo_root

    root = repo_root()
    listed = subprocess.run(
        ["git", "ls-files", "examples", "merlin/contract", "*.yaml"], cwd=root, capture_output=True, text=True
    ).stdout.split()
    return [root / p for p in listed if p.endswith((".yaml", ".yml")) and (root / p).stat().st_size < 4_000_000]


def test_the_fast_loader_reads_every_tracked_document_as_pyyaml_does():
    """libyaml scans and parses, PyYAML's safe constructor builds: the value must be PyYAML's own."""
    paths = _tracked_yaml()
    assert len(paths) > 50
    for path in paths:
        text = path.read_text(encoding="utf-8")
        try:
            want = yaml.safe_load(text)
        except yaml.YAMLError:
            continue
        assert Y._parse(text) == want, path


def test_a_malformed_document_raises_pyyamls_own_error():
    text = "a: [1, 2\nb: {"
    with pytest.raises(yaml.YAMLError) as theirs:
        yaml.safe_load(text)
    with pytest.raises(yaml.YAMLError) as ours:
        Y.safe_load_text(text)
    assert str(ours.value) == str(theirs.value)


def test_a_golden_document_is_parsed_once_and_each_reader_gets_its_own_copy(tmp_path, monkeypatch):
    from merlin.targetgen import golden_store as GS

    (tmp_path / "golden.yaml").write_text("golden_source: torch\noutputs: {Y0: [1, 2]}\n", encoding="utf-8")
    parses = []
    real = Y._parse
    monkeypatch.setattr(Y, "_parse", lambda text: parses.append(1) or real(text))
    first = GS.read_document(tmp_path)
    first["outputs"]["Y0"].append(3)
    assert GS.read_document(tmp_path) == {"golden_source": "torch", "outputs": {"Y0": [1, 2]}}
    assert len(parses) <= 1
