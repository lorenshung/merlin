"""Actual builder credit, falsifiability, and refusal tests for optional synthesis."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest
from merlin_experiments.phase0 import op_cell_synth as S
from merlin_experiments.phase0.sweeps import resolve_extent

from merlin.targetgen import op_form_regimes as F
from merlin.targetgen import opset_contract as O
from merlin.targetgen.compute_units import SemanticCapability
from merlin.targetgen.corpus_spec import CorpusBinding


def fixture(edge=8):
    b = F.FormBounds(
        "fixture", array_rows=F.Bound("array_rows", edge, "source"), array_cols=F.Bound("array_cols", edge, "source")
    )
    h = O.homes_for_capabilities({"contraction": SemanticCapability("contraction", dtypes=("int8",))}, ops=["matmul"])
    return (
        b,
        O.cell_coverage(h, b, [], capability_map={"contraction": SemanticCapability("contraction", dtypes=("int8",))}),
        CorpusBinding("fixture", edge, "int8", "i32", True, ["L1"], "exact_int"),
    )


def synth(b, r, binding, **kw):
    return S.synthesize_op_cells(
        "fixture",
        bounds=b,
        report=r,
        binding=binding,
        capability_map=kw.pop("capability_map", {"contraction": SemanticCapability("contraction", dtypes=("int8",))}),
        max_capsules=kw.pop("max_capsules", 20),
        **kw,
    )


@pytest.mark.parametrize("edge", [4, 8, 12])
def test_real_builder_closes_only_actual_cells_deterministically(edge):
    b, r, binding = fixture(edge)
    before = copy.deepcopy(r)
    result = synth(b, r, binding)
    assert result == synth(b, r, binding) and r == before
    assert result["capsules"]
    remaining = set(r["under"])
    for entry in result["capsules"]:
        refusal, cap, _ = S._build_probe(entry, binding)
        assert refusal is None
        remaining -= O.capsule_cells(cap, b)
    assert sorted(remaining) == result["provenance"]["still_uncovered"]
    assert all("UNDERIVABLE" in x for x in remaining)


def test_missing_binding_and_changed_tile_refuse():
    b, r, binding = fixture()
    with pytest.raises(ValueError):
        synth(b, r, None)
    binding = replace(binding, tile_dim=4)
    with pytest.raises(ValueError):
        synth(b, r, binding)


def test_budget_does_not_silently_truncate():
    with pytest.raises(ValueError, match="budget"):
        synth(*fixture(), max_capsules=1)


def test_fixed_geometry_builder_cannot_credit_proposed_shape(monkeypatch):
    b, r, binding = fixture()
    real = S._build_probe

    def fixed(entry, binding):
        return real(dict(entry, M=8, K=8, N=8), binding)

    monkeypatch.setattr(S, "_build_probe", fixed)
    result = synth(b, r, binding)
    assert "matmul/aspect=wide" in result["provenance"]["still_uncovered"]
    assert len(result["capsules"]) == 1


def test_renamed_builder_cannot_close_deferred_operation(monkeypatch):
    b, r, binding = fixture()
    real = S._build_probe

    def renamed(entry, binding):
        refusal, cap, size = real(entry, binding)
        cap["operation"]["op"] = "add"
        return refusal, cap, size

    monkeypatch.setattr(S, "_build_probe", renamed)
    result = synth(b, r, binding, skip_ops=["add"])
    assert not result["capsules"]
    assert result["provenance"]["cells_the_builder_refused"]


def test_unfalsifiable_golden_is_not_a_successful_build(monkeypatch):
    from merlin.targetgen import numeric_falsifiability as N

    def reject(*args, **kwargs):
        raise ValueError("independent golden is unfalsifiable")

    monkeypatch.setattr(N, "falsifiable_policy", reject)
    result = synth(*fixture())
    assert not result["capsules"]
    assert any("unfalsifiable" in x for x in result["provenance"]["cells_the_builder_refused"].values())


def test_certification_budget_uses_actual_built_output():
    result = synth(*fixture(), max_output_elements=64)
    assert any(e.get("max_oracle_tier") == "L2" for e in result["capsules"])
    assert any("max_oracle_tier" not in e for e in result["capsules"])


def test_skipped_operation_remains_visible():
    _, r, _ = fixture()
    result = synth(*fixture(), skip_ops=["matmul"])
    assert not result["capsules"]
    assert result["provenance"]["still_uncovered"] == r["under"]


def test_no_builder_or_unknown_axis_remains_uncovered():
    b, r, binding = fixture()
    key = "unimplemented/occupancy=aligned"
    r = {
        "cells": {key: {"op": "unimplemented", "family": "contraction", "axis": "occupancy", "regime": "aligned"}},
        "under": [key],
    }
    result = synth(b, r, binding)
    assert not result["capsules"] and result["provenance"]["still_uncovered"] == [key]


def test_extent_tokens_use_current_phase0_parser():
    for tile in (4, 8, 12):
        for value in range(1, 100):
            assert resolve_extent(S._tile_token(value, tile), tile) == value


def test_actual_shape_outside_capability_is_not_minted():
    caps = {"contraction": SemanticCapability("contraction", dtypes=("int8",), arbitrary_mnk=False)}
    result = synth(*fixture(), capability_map=caps)
    assert "matmul/occupancy=partial" in result["provenance"]["still_uncovered"]
    assert any("whole-tile" in reason for reason in result["provenance"]["cells_the_builder_refused"].values())


def test_actual_rank_refusal_not_just_family_membership():
    caps = {"contraction": SemanticCapability("contraction", dtypes=("int8",), ranks=(3,))}
    assert not synth(*fixture(), capability_map=caps)["capsules"]


def test_phase0_generator_keeps_private_access_identity():
    from merlin.common import access

    name = "merlin_experiments.phase0.op_cell_synth"
    assert any(access.module_matches(name, prefix) for prefix in access.declared_modules("grader"))


def _cli():
    import runpy

    from merlin.common.paths import repo_root

    namespace = runpy.run_path(str(repo_root() / "build_tools/scripts/synth_op_cell_capsules.py"))
    main = namespace["main"]
    return main, main.__globals__


def test_cli_refuses_existing_file_before_reading_inputs(tmp_path):
    main, _ = _cli()
    output = tmp_path / "retained.json"
    output.write_text("retained")
    with pytest.raises(ValueError, match="new file"):
        main(
            [
                "--input",
                str(tmp_path / "absent"),
                "--descriptor",
                str(tmp_path / "absent2"),
                "--output",
                str(output),
                "--max-capsules",
                "1",
            ]
        )
    assert output.read_text() == "retained"


def test_cli_source_corpus_guard_rejects_alias(tmp_path):
    from types import SimpleNamespace

    from merlin_experiments.phase0.generation import _require_distinct_corpus_destinations

    corpus = tmp_path / "source"
    corpus.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(corpus, target_is_directory=True)
    with pytest.raises(ValueError, match="overlaps source capsule corpus"):
        _require_distinct_corpus_destinations(
            SimpleNamespace(capsule_corpus=corpus), output_root=alias / "new.json", evidence_root=None
        )


def test_cli_rechecks_inputs_before_publication(tmp_path, monkeypatch):
    import hashlib
    from types import SimpleNamespace

    main, ns = _cli()
    inp = tmp_path / "input.json"
    inp.write_text("{}")
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text("target: fixture")
    b, r, binding = fixture()
    r["input"] = {"path": str(inp), "sha256": hashlib.sha256(inp.read_bytes()).hexdigest()}
    doc = {"target": "fixture", "datapath": {}, "contract": {}, "facts": {}}
    monkeypatch.setitem(ns, "read_request", lambda path: (doc, r, b))
    monkeypatch.setitem(ns, "load_target_experiment", lambda path: SimpleNamespace(target="fixture"))
    monkeypatch.setitem(ns, "derive_binding", lambda *args, **kw: binding)

    def mutate(*args, **kw):
        inp.write_text('{"mutated":true}')
        return {"capsules": [], "provenance": {}}

    monkeypatch.setitem(ns, "synthesize_op_cells", mutate)
    output = tmp_path / "new.json"
    with pytest.raises(ValueError, match="coverage input changed"):
        main(["--input", str(inp), "--descriptor", str(descriptor), "--output", str(output), "--max-capsules", "1"])
    assert not output.exists()
