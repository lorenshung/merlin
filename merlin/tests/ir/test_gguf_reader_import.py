"""An optional installed gguf-py remains usable without an external checkout."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from merlin.common.paths import ExternalPathUnset
from merlin.frontends import gguf_reader


def test_installed_gguf_is_usable_without_external_selection(monkeypatch, tmp_path):
    installed = ModuleType("gguf")
    installed.__file__ = str(tmp_path / "site-packages/gguf/__init__.py")
    monkeypatch.setitem(sys.modules, "gguf", installed)
    monkeypatch.setattr(
        gguf_reader, "ext_path", lambda _name: (_ for _ in ()).throw(ExternalPathUnset("unset"))
    )

    assert gguf_reader._gguf() is installed


def test_selected_checkout_refuses_installed_gguf_from_other_source(monkeypatch, tmp_path):
    selected = tmp_path / "llama.cpp"
    (selected / "gguf-py").mkdir(parents=True)
    installed = ModuleType("gguf")
    installed.__file__ = str(tmp_path / "site-packages/gguf/__init__.py")
    monkeypatch.setitem(sys.modules, "gguf", installed)
    monkeypatch.setattr(gguf_reader, "ext_path", lambda _name: selected)

    with pytest.raises(ImportError, match="outside MERLIN_EXT_LLAMA_CPP"):
        gguf_reader._gguf()


def test_selected_checkout_accepts_only_its_gguf_py(monkeypatch, tmp_path):
    selected = tmp_path / "llama.cpp"
    (selected / "gguf-py").mkdir(parents=True)
    module = ModuleType("gguf")
    module.__file__ = str(selected / "gguf-py/gguf/__init__.py")
    monkeypatch.setitem(sys.modules, "gguf", module)
    monkeypatch.setattr(gguf_reader, "ext_path", lambda _name: selected)

    assert gguf_reader._gguf() is module


def test_selected_checkout_requires_gguf_py(monkeypatch, tmp_path):
    monkeypatch.setattr(gguf_reader, "ext_path", lambda _name: tmp_path)
    with pytest.raises(FileNotFoundError, match="no gguf-py"):
        gguf_reader._gguf()
