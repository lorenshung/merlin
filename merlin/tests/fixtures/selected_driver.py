"""A target's whole-model driver modules, loaded from the EXPLICITLY selected support provider.

The driver (program generator, kernel binder, open-model dispatch) is target-owned and ships in the
OOT support package selected on ``MERLIN_TARGET_PATH``; no in-tree copy remains. A test of it skips
when no provider for the target is selected -- absence is not a passing qualification.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def require_support(target: str) -> Path:
    """The root of ``target``'s explicitly selected support provider, or skip when none is selected.

    Only ABSENCE skips: a provider that IS selected and then fails to load still fails the test that
    reaches it, so a backend broken by a refactor never reads as a green skip."""
    from merlin.targetgen import target_registry

    selected = target_registry.explicit_targets().get(target)
    if selected is None:
        pytest.skip(f"requires explicit {target} support on MERLIN_TARGET_PATH", allow_module_level=True)
    return Path(selected)


def driver_file(target: str, name: str) -> Path:
    """``<selected support>/whole_model/<name>`` for ``target``, or skip when none is selected."""
    from merlin.targetgen import target_registry

    require_support(target)
    path = Path(target_registry.resolve(target).base) / "whole_model" / name
    if not path.is_file():
        pytest.skip(f"the selected {target} support ships no whole_model/{name}", allow_module_level=True)
    return path


def load(target: str, name: str, *, module_name: str | None = None):
    """The driver module ``name`` of the selected provider, imported once under ``module_name``."""
    path = driver_file(target, name)
    key = module_name or f"selected_{target}_{path.stem}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module
