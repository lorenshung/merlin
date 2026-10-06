"""Linked device bytes must distinguish builds with identical host objects."""

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from merlin.runtime.backends.spike_model import _supplemental_object_digest


def test_changed_partial_linked_device_changes_identity_same_host(tmp_path):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("requires host C compiler")

    def compile_object(name, value):
        source = tmp_path / (name + ".c")
        obj = tmp_path / (name + ".o")
        source.write_text(f"int kernel(void) {{return {value};}}\n")
        subprocess.run([cc, "-c", str(source), "-o", str(obj)], check=True)
        merged = tmp_path / (name + "_catalog.o")
        subprocess.run([cc, "-r", str(obj), "-o", str(merged)], check=True)
        return merged

    a, b = compile_object("one", 1), compile_object("two", 2)
    host = b"identical model and weights"

    def identity(path):
        h = hashlib.sha256(host)
        h.update(_supplemental_object_digest([path]))
        return h.digest()

    assert identity(a) != identity(b)
    copied = tmp_path / "renamed.o"
    shutil.copyfile(a, copied)
    assert identity(a) == identity(copied)


def test_link_order_boundaries_and_missing_object_are_significant(tmp_path):
    paths = []
    for name, content in [("a", b"ab"), ("b", b"c"), ("c", b"a"), ("d", b"bc")]:
        p = tmp_path / name
        p.write_bytes(content)
        paths.append(p)
    assert _supplemental_object_digest(paths[:2]) != _supplemental_object_digest(paths[2:])
    assert _supplemental_object_digest(paths[:2]) != _supplemental_object_digest(paths[1::-1])
    with pytest.raises(FileNotFoundError):
        _supplemental_object_digest([tmp_path / "missing"])
