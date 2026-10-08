"""A lattice proof must describe the program and target actually lowered."""

from types import SimpleNamespace

import pytest

from merlin.verify import evaluate, lattice


def test_sweep_lowers_the_requested_target(monkeypatch):
    target = "selected_fixture"
    seen = []
    monkeypatch.setattr(
        lattice,
        "load_spec",
        lambda _: {
            "cells": [{"cell": "contraction/i8/aligned", "family": "contraction", "dtype": "i8"}],
            "boundaries": {"extent_probes": [{"boundary": "tile", "edge": 1, "source": "fixture", "points": [1]}]},
        },
    )

    def lower(m, k, n, reuse, *, target):
        seen.append(target)
        return object(), {"name": target}

    monkeypatch.setattr(evaluate, "_lower_to_interface", lower)
    monkeypatch.setattr(evaluate, "_finish_lowering", lambda iface, tc: {"target": tc["name"]})
    monkeypatch.setattr(
        "merlin.verify.refine.validate_compilation", lambda iface, cb, **kw: SimpleNamespace(status="unsat")
    )
    monkeypatch.setattr(lattice, "witnesses_declared", lambda _: {"capsules": 0, "distinct_shapes": 0})

    rec = lattice.sweep(target)
    assert seen == [target]
    assert rec["points_verified"] == 1
    assert rec["target"] == target
    assert rec["cells_covered"] == ["contraction/i8/aligned"]


def test_sweep_does_not_credit_i8_proof_to_another_dtype(monkeypatch):
    monkeypatch.setattr(
        lattice,
        "load_spec",
        lambda _: {
            "cells": [{"cell": "contraction/bf16/aligned", "family": "contraction", "dtype": "bf16"}],
            "boundaries": {"extent_probes": [{"boundary": "tile", "edge": 1, "source": "fixture", "points": [1]}]},
        },
    )

    def unexpected_lowering(*args, **kwargs):
        raise AssertionError("unsupported dtype was lowered")

    monkeypatch.setattr(evaluate, "_lower_to_interface", unexpected_lowering)
    monkeypatch.setattr(lattice, "witnesses_declared", lambda _: {"capsules": 0, "distinct_shapes": 0})

    rec = lattice.sweep("selected_fixture")
    assert rec["points_verified"] == 0
    assert rec["cells_covered"] == []
    assert rec["cell_omissions"][0]["cell"] == "contraction/bf16/aligned"
    assert "i8" in rec["cell_omissions"][0]["reason"]


def test_sampled_proofs_do_not_claim_full_lattice_coverage(monkeypatch):
    monkeypatch.setattr(
        lattice,
        "load_spec",
        lambda _: {
            "cells": [{"cell": "contraction/i8/aligned", "family": "contraction", "dtype": "i8"}],
            "boundaries": {"extent_probes": [{"boundary": "tile", "edge": 2, "source": "fixture", "points": [1, 2]}]},
        },
    )
    monkeypatch.setattr(evaluate, "_lower_to_interface", lambda *a, target, **kw: (object(), {"name": target}))
    monkeypatch.setattr(evaluate, "_finish_lowering", lambda iface, tc: {"target": tc["name"]})
    monkeypatch.setattr(
        "merlin.verify.refine.validate_compilation", lambda iface, cb, **kw: SimpleNamespace(status="unsat")
    )
    monkeypatch.setattr(lattice, "witnesses_declared", lambda _: {"capsules": 0, "distinct_shapes": 0})

    rec = lattice.sweep("selected_fixture", max_points=1)
    assert rec["cells_verified_at_sampled_points"] == ["contraction/i8/aligned"]
    assert rec["cells_covered"] == []
    assert "1 of 2 extent(s) selected" in lattice.render(rec)


def test_unexpected_lowering_error_is_visible_and_not_credited(monkeypatch):
    monkeypatch.setattr(
        lattice,
        "load_spec",
        lambda _: {
            "cells": [{"cell": "contraction/i8/aligned", "family": "contraction", "dtype": "i8"}],
            "boundaries": {"extent_probes": [{"boundary": "tile", "edge": 1, "source": "fixture", "points": [1]}]},
        },
    )

    def broken_lowering(*args, **kwargs):
        raise RuntimeError("broken tool")

    monkeypatch.setattr(evaluate, "_lower_to_interface", broken_lowering)
    monkeypatch.setattr(lattice, "witnesses_declared", lambda _: {"capsules": 0, "distinct_shapes": 0})

    rec = lattice.sweep("selected_fixture")
    assert rec["points_error"] == 1
    assert rec["cells_covered"] == []
    assert "broken tool" in lattice.render(rec)


def test_dynamic_shape_count_uses_only_selected_target_roots(monkeypatch, tmp_path):
    from merlin.targetgen import corpora

    selected = tmp_path / "selected"
    other = tmp_path / "other"
    for root, shape in ((selected, [2, 2]), (other, [9, 9])):
        capsule = root / "case"
        capsule.mkdir(parents=True)
        (capsule / "capsule.yaml").write_text(f"inputs:\n  - shape: {shape}\n", encoding="utf-8")
    monkeypatch.setattr(corpora, "graded_capsule_roots", lambda target: [selected] if target == "selected" else [other])

    rec = lattice.witnesses_declared("selected")
    assert rec["capsules"] == 1
    assert rec["distinct_shapes"] == 1


def test_selected_contract_must_name_requested_target(monkeypatch):
    from merlin.xdsl_dialects.lowering import pipeline

    monkeypatch.setattr(pipeline, "load_curated_contract", lambda target: {"name": "other_fixture"})
    with pytest.raises(ValueError, match="expected 'selected_fixture'"):
        evaluate._lower_to_interface(1, 1, 1, 2, target="selected_fixture")


def test_an_unreadable_declared_capsule_is_reported_not_dropped(monkeypatch, tmp_path):
    from merlin.targetgen import corpora

    root = tmp_path / "selected"
    (root / "good").mkdir(parents=True)
    (root / "good" / "capsule.yaml").write_text("inputs:\n  - shape: [2, 2]\n", encoding="utf-8")
    (root / "bad").mkdir()
    (root / "bad" / "capsule.yaml").write_text("inputs: [unterminated\n", encoding="utf-8")
    monkeypatch.setattr(corpora, "graded_capsule_roots", lambda target: [root])

    rec = lattice.witnesses_declared("selected")
    assert rec["capsules"] == 1
    assert [row["path"] for row in rec["unreadable"]] == [str(root / "bad" / "capsule.yaml")]
