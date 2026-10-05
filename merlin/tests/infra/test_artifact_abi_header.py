"""A registered device states the parameter-header ABI its elaboration takes, so a program built for
another header is refused before a cycle is spent -- never run and read as a schedule regression."""

from __future__ import annotations

import yaml

from merlin.common import provenance as P


def _registry(tmp_path, **body):
    path = tmp_path / "pins.yaml"
    path.write_text(yaml.safe_dump({"pins": {}, "artifacts": {"emu": {"path": "/x", "digest": "d" * 64, **body}}}))
    return path


def test_an_artifact_carries_its_declared_abi_header(tmp_path):
    artifact = P.load_artifacts(_registry(tmp_path, abi_header_sha256="a" * 64))["emu"]
    assert artifact.abi_header_sha256 == "a" * 64


def test_an_artifact_without_one_is_unknown_never_assumed(tmp_path):
    assert P.load_artifacts(_registry(tmp_path))["emu"].abi_header_sha256 == ""


def test_the_registry_states_the_abi_of_every_device_a_whole_model_is_measured_on():
    artifacts = P.load_artifacts()
    for name, artifact in artifacts.items():
        if artifact.role in ("gsim_binary",) and artifact.target and artifact.digest:
            assert len(artifact.abi_header_sha256) in (0, 64), name
    boards = [a for a in artifacts.values() if a.role == "firesim_bitstream" and a.abi_header_sha256]
    assert boards and all(len(a.abi_header_sha256) == 64 and a.hw_configs for a in boards)
