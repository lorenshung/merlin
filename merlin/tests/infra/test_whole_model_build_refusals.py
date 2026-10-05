"""Two silent degradations of a whole-model build, each refused by name instead:

* RTL facts with no datapath: grouping could place no epilogue on the device, so a closed model would
  read as an OPEN one (host code between every group, about twice the groups);
* a package build in which the package answers no group: its program is the library's, and every
  cycle would be credited to a package that did nothing.
"""

from __future__ import annotations

import pytest

from merlin.perf import whole_model_build as W
from merlin.perf import whole_model_open_service as OS
from merlin.targetgen.rtl import facts as F


@pytest.mark.parametrize(
    "load",
    [
        lambda target: (_ for _ in ()).throw(FileNotFoundError("no artifact")),
        lambda target: {"facts": {"target": target, "arrays": [{"rows": 4}]}},
    ],
    ids=["no-facts", "no-datapath"],
)
def test_missing_datapath_facts_refuse_and_never_produce_an_open_model(monkeypatch, load):
    from merlin.xdsl_dialects.lowering import compute_groups as CG

    monkeypatch.setattr(F, "load_facts", load)
    monkeypatch.setattr(CG, "form_groups", lambda *a, **k: pytest.fail("grouped a model without its facts"))
    with pytest.raises(W.WholeModelBuildError, match="no datapath facts for toy"):
        W.require_datapath_facts("toy")
    with pytest.raises(W.WholeModelBuildError, match="no datapath facts for toy"):
        OS.is_open_model("/nonexistent/capsule", "toy")


def test_facts_with_a_datapath_are_returned():
    doc = {"facts": {"datapaths": [{"name": "input", "dtype": "i8"}]}}
    import merlin.targetgen.rtl.facts as facts

    original = facts.load_facts
    try:
        facts.load_facts = lambda target: doc
        assert W.require_datapath_facts("toy") is doc
    finally:
        facts.load_facts = original


DECLINED = {
    "on": W.ON_HOST,
    "cause": "library_loop_free_path",
    "declined_as": "interface_capsule_failed",
    "declined_why": "interface capsule: ValueError: command target must declare non-empty encoding.corpus_issue_order",
}


def test_a_package_build_that_answers_no_group_is_refused_with_its_reason():
    rows = [{"group": 0, "on": W.ON_HOST}] + [{"group": g, **DECLINED} for g in range(1, 71)]
    with pytest.raises(W.WholeModelBuildError, match="answered none of 71 group") as caught:
        W.refuse_if_every_group_declined("/pkg", rows)
    assert "70 x interface_capsule_failed" in str(caught.value) and "corpus_issue_order" in str(caught.value)


def test_one_answered_group_a_library_build_or_the_callers_own_decline_is_not_refused():
    rows = [{"group": g, **DECLINED} for g in range(1, 4)]
    W.refuse_if_every_group_declined("/pkg", [*rows, {"group": 4, "on": W.ON_PACKAGE}])
    W.refuse_if_every_group_declined(None, rows)  # the target's library program: nothing to credit
    W.refuse_if_every_group_declined("/pkg", [{"group": 1, "on": W.ON_VENDOR, "cause": "caller_declined"}])
