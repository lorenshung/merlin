"""whole_model_builder.build forwards `decline` to the underlying build (deliverable: cell mode's
"give a session one cell" wiring). Empty by default, so an ordinary whole-model build call is
unaffected; a cell session's decline list (every group outside the cell) now reaches the same
mechanism the measured mode's worker already reads `options.get("decline")` for.
"""

from __future__ import annotations

import pytest

from merlin.perf import whole_model_build as WMB
from merlin.perf import whole_model_builder as builder
from merlin.perf import whole_model_open as WO

pytestmark = pytest.mark.target("gemmini")


class _Captured(Exception):
    def __init__(self, kwargs):
        self.kwargs = kwargs
        super().__init__("captured")


def test_an_explicit_decline_list_reaches_the_underlying_build(monkeypatch):
    def fake_build(package_dir, model_capsule, **kwargs):
        raise _Captured(kwargs)

    monkeypatch.setattr(WMB, "build", fake_build)
    # A closed model: the builder routes an OPEN one to the open-model builder before any of this.
    monkeypatch.setattr(WO, "is_open_model", lambda capsule, target: False)
    with pytest.raises(_Captured) as excinfo:
        builder.build(
            "unused-package",
            target="gemmini",
            out_dir="unused-out",
            model_capsule="unused-capsule",
            machine="unused-machine",
            header="unused-header.h",
            decline=["residual_add", "g33"],
        )
    assert excinfo.value.kwargs["decline"] == ["residual_add", "g33"]


def test_no_decline_argument_forwards_an_empty_list_never_none(monkeypatch):
    def fake_build(package_dir, model_capsule, **kwargs):
        raise _Captured(kwargs)

    monkeypatch.setattr(WMB, "build", fake_build)
    # A closed model: the builder routes an OPEN one to the open-model builder before any of this.
    monkeypatch.setattr(WO, "is_open_model", lambda capsule, target: False)
    with pytest.raises(_Captured) as excinfo:
        builder.build(
            "unused-package",
            target="gemmini",
            out_dir="unused-out",
            model_capsule="unused-capsule",
            machine="unused-machine",
            header="unused-header.h",
        )
    assert excinfo.value.kwargs["decline"] == ()


def test_allow_passes_and_allow_regions_reach_the_underlying_build_and_default_off(monkeypatch):
    def fake_build(package_dir, model_capsule, **kwargs):
        raise _Captured(kwargs)

    monkeypatch.setattr(WMB, "build", fake_build)
    # A closed model: the builder routes an OPEN one to the open-model builder before any of this.
    monkeypatch.setattr(WO, "is_open_model", lambda capsule, target: False)
    with pytest.raises(_Captured) as excinfo:
        builder.build(
            "unused-package",
            target="gemmini",
            out_dir="unused-out",
            model_capsule="unused-capsule",
            machine="unused-machine",
            header="unused-header.h",
        )
    assert excinfo.value.kwargs["allow_passes"] is False and excinfo.value.kwargs["allow_regions"] is False

    with pytest.raises(_Captured) as excinfo:
        builder.build(
            "unused-package",
            target="gemmini",
            out_dir="unused-out",
            model_capsule="unused-capsule",
            machine="unused-machine",
            header="unused-header.h",
            allow_passes=True,
            allow_regions=True,
        )
    assert excinfo.value.kwargs["allow_passes"] is True and excinfo.value.kwargs["allow_regions"] is True
