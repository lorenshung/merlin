"""Portable explicit inputs retain bytes and executable intent, not source aliases."""

from __future__ import annotations

import importlib.util
import io
import tarfile

import pytest

from merlin.common.paths import repo_root


@pytest.fixture
def delivery(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    script = repo_root() / "build_tools/scripts/package_worker_inputs.py"
    spec = importlib.util.spec_from_file_location("worker_input_delivery", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pack_verify_extract_is_private_deterministic_and_source_preserving(delivery, tmp_path):
    source = tmp_path / "toolchain"
    (source / "bin").mkdir(parents=True)
    binary = source / "bin/compiler"
    binary.write_bytes(b"selected executable\n")
    binary.chmod(0o555)
    (source / "bin/cc").symlink_to("compiler")
    (source / "compiler").symlink_to("bin/compiler")
    (source / "bin-alias").symlink_to("bin", target_is_directory=True)
    data = source / "weights"
    data.write_bytes(bytes(range(256)) * 8192)
    data.chmod(0o400)
    original = {p: (p.lstat().st_mode, p.lstat().st_ino) for p in (source, binary, data)}
    first = delivery.pack({"toolchain": source}, provenance={"role": "selected tools"})
    second = delivery.pack({"toolchain": source}, provenance={"role": "selected tools"})
    report = delivery.verify(first / "worker-inputs.tar")
    assert report == delivery.verify(second / "worker-inputs.tar")
    assert report["payload_bytes"] == binary.stat().st_size + data.stat().st_size
    destination = tmp_path / "worker"
    delivery.verify(first / "worker-inputs.tar", expected_sha256=report["archive_sha256"], extract=destination)
    assert (destination / "inputs/toolchain/compiler").read_bytes() == binary.read_bytes()
    assert (destination / "inputs/toolchain/bin/compiler").stat().st_mode & 0o777 == 0o700
    assert (destination / "inputs/toolchain/weights").stat().st_mode & 0o777 == 0o600
    assert (destination / "inputs/toolchain/weights").stat().st_ino != data.stat().st_ino
    assert {p: (p.lstat().st_mode, p.lstat().st_ino) for p in original} == original
    assert first.stat().st_mode & 0o777 == destination.stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o077 == 0 for p in first.iterdir())
    with pytest.raises(ValueError, match="destination must be new"):
        delivery.verify(first / "worker-inputs.tar", extract=destination)


def test_links_fail_closed_and_output_cannot_be_inside_source(delivery, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "secret"
    outside.write_bytes(b"never implicitly read")
    link = source / "alias"
    for value in (str(outside), "../secret", "alias", "."):
        link.symlink_to(value)
        with pytest.raises(ValueError, match="relative links|escapes|cyclic"):
            delivery.pack({"input": source})
        link.unlink()
    link.symlink_to("missing")
    with pytest.raises(ValueError, match="broken"):
        delivery.pack({"input": source})
    with pytest.raises(ValueError, match="output cannot be inside"):
        delivery.pack({"input": tmp_path})


def _rewrite(original, replacement, transform):
    with tarfile.open(original) as incoming, tarfile.open(replacement, "w", format=tarfile.PAX_FORMAT) as outgoing:
        for info in incoming:
            payload = incoming.extractfile(info).read() if info.isfile() else None
            info, payload = transform(info, payload)
            outgoing.addfile(info, io.BytesIO(payload) if payload is not None else None)


def test_mutation_extra_members_traversal_and_wrong_digest_are_refused(delivery, tmp_path):
    source = tmp_path / "input"
    source.write_bytes(b"correct input bytes")
    product = delivery.pack({"selected": source})
    archive = product / "worker-inputs.tar"
    bad = tmp_path / "bad.tar"

    def mutation(info, payload):
        if info.name == "inputs/selected":
            payload = b"X" + payload[1:]
        return info, payload

    _rewrite(archive, bad, mutation)
    with pytest.raises(ValueError, match="payload digest"):
        delivery.verify(bad)
    with pytest.raises(ValueError, match="expected identity"):
        delivery.verify(archive, expected_sha256="0" * 64)

    def traversal(info, payload):
        if info.name == "inputs/selected":
            info.name = "../escaped"
        return info, payload

    _rewrite(archive, bad, traversal)
    with pytest.raises(ValueError, match="differs from manifest"):
        delivery.verify(bad, extract=tmp_path / "new")
    assert not (tmp_path / "escaped").exists()
    _rewrite(archive, bad, lambda info, payload: (info, payload))
    with bad.open("ab") as stream:
        stream.write(b"unlisted trailing bytes")
    with pytest.raises(ValueError, match="archive terminator|trailing bytes"):
        delivery.verify(bad)


def test_source_changes_during_packing_leave_no_accepted_bundle(delivery, tmp_path, monkeypatch):
    source = tmp_path / "input"
    source.write_bytes(b"before")
    original = delivery.tarfile.TarFile.addfile

    def changing(self, info, fileobj=None):
        result = original(self, info, fileobj)
        if info.name == "inputs/selected":
            source.write_bytes(b"after!")
        return result

    monkeypatch.setattr(delivery.tarfile.TarFile, "addfile", changing)
    with pytest.raises(ValueError, match="source inputs changed"):
        delivery.pack({"selected": source})


def test_preexisting_parent_symlinks_are_refused(delivery, tmp_path):
    source = tmp_path / "input"
    source.write_bytes(b"selected bytes")
    product = delivery.pack({"selected": source})
    alias = tmp_path / "parent-alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="ancestry must be real"):
        delivery.verify(product / "worker-inputs.tar", extract=alias / "worker")
    assert not (tmp_path / "worker").exists()
    with pytest.raises(ValueError, match="ancestry must be real"):
        delivery.pack({"selected": alias / "input"})
