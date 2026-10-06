"""Skip a test whose SUBJECT is an external, machine-specific checkout that this host does not have.

Hardware sources (a Chipyard tree, a target's RTL fork, a vendor simulator) are never in this repo;
they are named by ``MERLIN_EXT_<NAME>`` in the environment or the gitignored ``.env`` and resolved with
:func:`merlin.common.paths.ext_path`. A hosted CI runner has none of them, so a test that reads one
must skip there rather than fail on a ``KeyError`` from deep inside the library.

Only ABSENCE skips, mirroring :func:`selected_driver.require_support`: a name that is unset, or set to a
path that does not exist, is absence. A checkout that IS present and then fails still fails the test
that reaches it, so an extraction broken by a refactor never reads as a green skip.
"""

from __future__ import annotations

import pytest


def missing(*names: str) -> list[str]:
    """The ``MERLIN_EXT_<NAME>`` keys among ``names`` that are unset or point at nothing."""
    from merlin.common.paths import ext_path

    absent = []
    for name in names:
        try:
            present = ext_path(name).exists()
        except KeyError:
            present = False
        if not present:
            absent.append(f"MERLIN_EXT_{name.upper()}")
    return absent


def requires_ext(*names: str):
    """A ``skipif`` marker for a test or module that reads the named external checkouts."""
    absent = missing(*names)
    return pytest.mark.skipif(bool(absent), reason=f"external checkout(s) not available: {', '.join(absent)}")


def require_ext(*names: str) -> None:
    """Skip the calling test (or, at import time, its module) when a named checkout is absent."""
    absent = missing(*names)
    if absent:
        pytest.skip(f"external checkout(s) not available: {', '.join(absent)}", allow_module_level=True)
