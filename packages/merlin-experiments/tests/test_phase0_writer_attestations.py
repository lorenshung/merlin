"""A shared capture adapter must not lend another capsule its execution evidence."""

from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase0 import sealed_generation, writer

from merlin.targetgen import capsule_source


@pytest.mark.parametrize("kind", ["model", "pytorch"])
def test_shared_adapter_records_only_this_capsules_captures(tmp_path, monkeypatch, kind):
    source = SimpleNamespace(available=lambda: True, attestations=[{"unrelated": "older capsule"}])
    binding = SimpleNamespace(target="test", operand_dtype="f32")
    monkeypatch.setattr(writer.CS, "entry_binding", lambda *_: ("simt", binding))

    def write(entry, _binding, output, *, source):
        directory = output / entry["name"]
        directory.mkdir()
        (directory / "capsule.yaml").write_text(yaml.safe_dump({"source": "pytorch"}))
        for index in range(entry["captures"]):
            source.attestations.append({"capsule": entry["name"], "capture": index})
        if entry.get("fail"):
            raise ValueError("capture succeeded but capsule packaging failed")
        return directory

    monkeypatch.setattr(capsule_source, f"write_{kind}_capsule", write)
    base = {"kind": "model"} if kind == "model" else {"source": "pytorch"}
    for name, count in (("first", 2), ("second", 1), ("uncaptured", 0)):
        entry = {**base, "name": name, "captures": count}
        written = writer._write_capsule_inner(entry, binding, tmp_path, capture=source)
        capsule = yaml.safe_load((written / "capsule.yaml").read_bytes())
        expected = [{"capsule": name, "capture": index} for index in range(count)]
        assert capsule.get("capture_execution_attestations", []) == expected
        if not count:
            assert sealed_generation.verified_capture_failure(capsule) == (
                "generation-time capture carries no sealed-runner attestation"
            )

    # The adapter can retain its complete audit history; only the capsule's
    # provenance is scoped. An unsuccessful write cannot lend its evidence either.
    with pytest.raises(ValueError, match="packaging failed"):
        writer._write_capsule_inner(
            {**base, "name": "failed", "captures": 1, "fail": True}, binding, tmp_path, capture=source
        )
    written = writer._write_capsule_inner(
        {**base, "name": "after_failure", "captures": 1}, binding, tmp_path, capture=source
    )
    assert yaml.safe_load((written / "capsule.yaml").read_bytes())["capture_execution_attestations"] == [
        {"capsule": "after_failure", "capture": 0}
    ]
    assert len(source.attestations) == 6
