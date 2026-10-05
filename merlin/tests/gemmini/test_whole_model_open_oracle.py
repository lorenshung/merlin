"""The open model's oracle, computed in its own process or read from its cache, is the in-build one."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from merlin.common import mlir_query as mq
from merlin.perf import whole_model_open as WO
from merlin.perf import whole_model_open_oracle as OO
from merlin.runtime.backends import base as backends
from merlin.xdsl_dialects.lowering import compute_groups as CG


def _open_test_module():
    import sys
    from pathlib import Path

    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    try:
        return importlib.import_module("test_whole_model_open")
    finally:
        sys.path.remove(str(here))


@pytest.fixture()
def model(tmp_path, monkeypatch):
    W = _open_test_module()
    interface = tmp_path / "capsule.interface.mlir"
    interface.write_text(W.MODEL, encoding="utf-8")
    arguments = list(W._inputs())
    monkeypatch.setattr(WO, "forward_arguments", lambda capsule, extra=None: (arguments, {}))
    return SimpleNamespace(interface=interface, directory=tmp_path), arguments, W._TARGET


def _in_build(capsule, arguments, target):
    """The oracle exactly as the build computed it before the job: over the cut's own dispatches."""
    from merlin.frontends.linalg_mlir import parse_mlir_file

    original = mq.parse(capsule.interface.read_text(encoding="utf-8"))
    groups = CG.form_groups(original, target)
    by_region = {WO._region_of(g): int(g.index) for g in groups if g.placement != CG.HOST}
    module = parse_mlir_file(capsule.interface)
    WO.normalize(module)
    import dataclasses

    device = [
        dataclasses.replace(g, index=by_region[WO._region_of(g)])
        for g in CG.form_groups(module, target)
        if g.placement != CG.HOST
    ]
    _main, _host, dispatches = WO.externalize_dispatches(module, device)
    digest = backends.whole_model_driver(target).program.group_digest
    return WO.oracle(original, arguments, dispatches, groups, group_digest=digest), [d.group for d in dispatches]


def _same(a, b):
    assert len(a["outputs"]) == len(b["outputs"])
    for x, y in zip(a["outputs"], b["outputs"], strict=True):
        assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
    assert list(a["dispatch"].items()) == list(b["dispatch"].items())


def test_the_job_computes_the_in_build_oracle_for_the_cuts_dispatches(model):
    capsule, arguments, target = model
    want, groups = _in_build(capsule, arguments, target)
    got, indices = OO.compute(capsule, target=target)
    assert indices == groups
    _same(got, want)


def test_a_recorded_oracle_reads_back_byte_for_byte_and_only_for_its_dispatches(model, tmp_path):
    capsule, arguments, target = model
    want, groups = _in_build(capsule, arguments, target)
    OO._write(tmp_path / "entry", want, groups)
    job = object.__new__(OO.OpenOracleJob)
    job._process, job.entry, job.state = None, tmp_path / "entry", "hit"
    _same(job.result(groups), want)
    with pytest.raises(WO.OpenModelError, match="another set of device dispatches"):
        job.result([*groups, 10_000])


def test_an_incomplete_record_is_not_a_hit(tmp_path):
    (tmp_path / "entry").mkdir()
    (tmp_path / "entry" / "dispatch.json").write_text("{}", encoding="utf-8")
    assert OO._read(tmp_path / "entry") is None
