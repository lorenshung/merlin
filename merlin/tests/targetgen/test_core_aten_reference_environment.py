"""Source references use the declared capture interpreter, independently of compiler Python."""

import os
from types import SimpleNamespace

import pytest

from merlin.targetgen import core_aten_numeric


@pytest.mark.parametrize("arguments", [(), ("--native",)])
def test_source_reference_uses_declared_capture_python(tmp_path, monkeypatch, arguments):
    from merlin.integrations import model2mlir
    from merlin.common.paths import python_import_roots

    python = tmp_path / "capture-python"
    monkeypatch.setattr(model2mlir, "capture_python", lambda: python)
    monkeypatch.setenv("MERLIN_COMPILER_PYTHON", "/unavailable/compiler-python")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(core_aten_numeric.subprocess, "run", run)
    assert core_aten_numeric._source_reference(tmp_path, *arguments).returncode == 0
    command, kwargs = calls.pop()
    assert command == [str(python), "-m", "merlin.targetgen.core_aten_w8a8_reference", str(tmp_path), *arguments]
    assert kwargs["env"]["PYTHONPATH"] == os.pathsep.join(str(p) for p in python_import_roots())
    assert not calls
