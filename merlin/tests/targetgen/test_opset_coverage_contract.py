"""Retained coverage/synthesis behavior with explicit, independent toy contracts."""

from __future__ import annotations

import copy
import json
import subprocess
import sys

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen import op_form_regimes as F
from merlin.targetgen import opset_contract as O
from merlin.targetgen import semantic_families as S
from merlin.targetgen.compute_units import SemanticCapability


def bounds(edge=8):
    return F.FormBounds(
        "fixture",
        array_rows=F.Bound("array_rows", edge, "source"),
        array_cols=F.Bound("array_cols", edge, "source"),
        operand_rows=F.Bound("operand_rows", 16, "source"),
        accumulator_rows=F.Bound("accumulator_rows", 8, "source"),
        dma_max_bytes=F.Bound("dma_max_bytes", 64, "source"),
        element_bits=F.Bound("element_bits", 8, "source"),
    )


def capsule(m=8, k=8, n=8, name="public"):
    return {
        "name": name,
        "label": "public",
        "inputs": [
            {"name": "W", "role": "weight", "shape": [k, n], "dtype": "i8"},
            {"name": "A", "role": "input", "shape": [m, k], "dtype": "i8"},
        ],
        "operation": {"op": "matmul", "attributes": {"lhs": "A", "weight": "W"}},
    }


def capmap():
    return {"contraction": SemanticCapability("contraction", dtypes=("int8",))}


def homes():
    return O.homes_for_capabilities(
        {"contraction": SemanticCapability("contraction", dtypes=("int8",))}, ops=("matmul",)
    )


def test_vocabulary_resource_and_current_semantics_preserved():
    assert "residual_add" in O.frontend_opset()
    assert S.check() == []
    assert S.operation_form("permute") == "permutation"
    assert S.from_op("reduce_mean") == "reduction"
    assert S.from_op("matmul_batched") == "contraction"
    assert set(S.form_axes("contraction")) <= set(S.form_axes("attention"))
    assert O.home_of("model", cap_map={}).home == O.STRUCTURAL
    assert O.home_of("not_declared", cap_map={}).home == O.UNKNOWN


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_bounds_refuse_invalid_source_facts(value):
    with pytest.raises(ValueError):
        F.Bound("array_rows", value, "fact")


def test_missing_bounds_are_visible_not_one_regime():
    r = F.regimes_for_family("contraction", F.FormBounds("absent"))
    assert r and all(x.regime == F.UNDERIVABLE for x in r)
    with pytest.raises(ValueError):
        F.Bound("array_rows", None, "")


@pytest.mark.parametrize(
    "m,k,n,aspect,depth",
    [(8, 8, 24, "wide", "single_pass"), (24, 24, 8, "tall_thin", "multi_pass"), (8, 8, 8, "square", "single_pass")],
)
def test_matrix_roles_not_input_position(m, k, n, aspect, depth):
    c = capsule(m, k, n)
    r = F.classify_capsule(c, bounds())
    assert r["aspect"] == aspect and r["reduction_depth"] == depth
    c["inputs"].reverse()
    assert F.classify_capsule(c, bounds()) == r


@pytest.mark.parametrize("change", [{"transpose_b": True}, {"kh": 3}, {"lhs": "missing"}])
def test_unproved_matrix_mapping_refuses(change):
    c = capsule()
    c["operation"]["attributes"].update(change)
    assert F.classify_capsule(c, bounds())["aspect"] == F.UNDERIVABLE


def test_incompatible_k_and_ambiguous_roles_refuse():
    c = capsule()
    c["inputs"][0]["shape"][0] = 12
    assert F.classify_capsule(c, bounds())["aspect"] == F.UNDERIVABLE
    c = capsule()
    c["operation"]["attributes"] = {}
    c["inputs"].append(dict(c["inputs"][1], name="B"))
    assert F.classify_capsule(c, bounds())["aspect"] == F.UNDERIVABLE


def test_capacity_requires_layout_not_input_size_and_transfer_not_max_extent():
    c = capsule(m=1024)
    r = F.classify_capsule(c, bounds())
    assert all(r[a] == F.UNDERIVABLE for a in ("operand_capacity", "accumulator_capacity", "transfer"))
    c["form_evidence"] = {
        "operand_rows": {"value": 17, "source": "layout"},
        "accumulator_rows": {"value": 8, "source": "layout"},
        "max_contiguous_bytes": {"value": 32, "source": "layout"},
    }
    r = F.classify_capsule(c, bounds())
    assert r["operand_capacity"] == "spills" and r["accumulator_capacity"] == "fits" and r["transfer"] == "single"
    c["form_evidence"]["max_contiguous_bytes"]["value"] = 65
    assert F.classify_capsule(c, bounds())["transfer"] == "split"


def test_rectangular_array_does_not_inherit_square_mapping():
    b = bounds()
    b.array_rows = F.Bound("array_rows", 4, "fact")
    assert F.classify_capsule(capsule(), b)["occupancy"] == F.UNDERIVABLE
    assert F.smallest_shape_for("occupancy", "aligned", b) is None


@pytest.mark.parametrize("label", ["hidden", "holdout", None])
def test_public_cohort_refuses_hidden_and_unknown(label):
    c = capsule()
    c["label"] = label
    with pytest.raises(ValueError):
        O.cell_coverage(homes(), bounds(), [c], capability_map=capmap())


def test_duplicate_names_and_stale_exclusions_refuse():
    with pytest.raises(ValueError):
        O.public_cohort([capsule(), capsule()])
    with pytest.raises(ValueError):
        O.public_cohort([capsule()], excluded=["absent"])


def test_demand_excludes_members_counts_epilogues_and_does_not_mutate():
    a = capsule(name="keep")
    a["operation"]["attributes"]["epilogue"] = ["bias_add"]
    b = capsule(name="drop")
    before = copy.deepcopy([a, b])
    d = O.demanded_ops([a, b], excluded=["drop"])
    assert d["matmul"].total == 1 and d["bias_add"].as_epilogue == 1
    assert [a, b] == before


def test_unknown_cells_cannot_be_covered_and_stale_debt_fails():
    report = O.cell_coverage(homes(), F.FormBounds("fixture"), [capsule()], capability_map=capmap())
    assert report["under"] and all("UNDERIVABLE" in key for key in report["under"])
    debt = {f"cell:{key}" for key in report["under"]}
    assert O.problems(report, ratchet=debt) == []
    assert O.problems(report, ratchet=debt | {"old"}) == ["stale:old"]


def test_union_axes_not_cartesian_and_epilogue_obligation():
    report = O.cell_coverage(homes(), bounds(), [capsule()], capability_map=capmap(), epilogue_stages=["bias_add"])
    assert "matmul/epilogue=bias_add" in report["under"]
    assert "matmul/occupancy=aligned" not in report["under"]
    assert len(report["cells"]) < 20


def test_gate_explicit_inputs_and_no_default_target_discovery():
    p = subprocess.run(
        [sys.executable, str(repo_root() / "build_tools/scripts/check_opset_coverage.py")],
        capture_output=True,
        text=True,
    )
    assert p.returncode == 2 and "--input" in p.stderr


def test_empty_or_malformed_universe_refuses(tmp_path, monkeypatch):
    p = tmp_path / "schema.json"
    p.write_text(json.dumps({"properties": {"operation": {"properties": {"op": {"enum": []}}}}}))
    monkeypatch.setattr(O, "data_path", lambda *args: p)
    with pytest.raises(ValueError):
        O.frontend_opset()


def test_declared_family_without_formats_is_unknown():
    assert O.home_of("matmul", cap_map={"contraction": SemanticCapability("contraction")}).home == O.UNKNOWN


def test_bad_member_format_is_not_coverage():
    c = capsule()
    c["inputs"][0]["dtype"] = "f32"
    report = O.cell_coverage(homes(), bounds(), [c], capability_map=capmap())
    assert report["refused_members"] and "member:public" in O.problems(report)
    assert "matmul/occupancy=aligned" in report["under"]


def test_array_store_facts_are_derived_and_nonbyte_format_not_truncated(monkeypatch):
    from types import SimpleNamespace

    from merlin.targetgen import address_space as A

    space = SimpleNamespace(array_name="grid", array_rows=6, array_cols=6)
    store = SimpleNamespace(total_rows=12, row_elems=6, row_bytes=3, element_bits=4)
    monkeypatch.setattr(A, "derive_address_space", lambda target, facts: space)
    monkeypatch.setattr(A, "operand_store", lambda space, dtype: SimpleNamespace(store=store, basis="fact"))
    monkeypatch.setattr(A, "accumulator_store", lambda space: SimpleNamespace(store=store, basis="fact"))
    b = F.bounds_for_target("fixture", facts={}, contract={"memory_model": {"dma": {"max_transfer_bytes": 9}}})
    assert b.array_rows.value == 6 and b.element_bits.value == 4 and b.dma_max_bytes.value == 9


def test_bad_dma_declaration_is_unknown(monkeypatch):
    from types import SimpleNamespace

    from merlin.targetgen import address_space as A

    monkeypatch.setattr(
        A,
        "derive_address_space",
        lambda target, facts: SimpleNamespace(array_name=None, array_rows=None, array_cols=None),
    )
    monkeypatch.setattr(A, "operand_store", lambda *args, **kw: SimpleNamespace(store=None, reason="absent"))
    monkeypatch.setattr(A, "accumulator_store", lambda *args: SimpleNamespace(store=None, reason="absent"))
    b = F.bounds_for_target("fixture", facts={}, contract={"memory_model": {"dma": {"max_transfer_bytes": True}}})
    assert not b.dma_max_bytes.derived


def test_input_request_real_fact_and_capability_adapter(tmp_path):
    from merlin.targetgen.opset_coverage_input import read_request

    contract = {
        "compute_units": [
            {
                "name": "array",
                "kind": "systolic",
                "dtypes": ["int8"],
                "semantic_capabilities": [{"family": "contraction", "dtypes": ["int8"]}],
            }
        ]
    }
    doc = {
        "schema": "merlin.opset_coverage_input.v1",
        "target": "fixture",
        "contract": contract,
        "facts": {"facts": {"arrays": [{"name": "grid", "rows": 8, "cols": 8}]}},
        "capsules": [capsule()],
    }
    path = tmp_path / "request.json"
    path.write_text(json.dumps(doc))
    _, report, b = read_request(path)
    assert b.tile_edge == 8 and "matmul/occupancy=aligned" not in report["under"]
    assert report["input"]["sha256"]
    doc["excluded"] = "public"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="unique nonempty names"):
        read_request(path)


def test_epilogue_string_cannot_be_counted_as_characters():
    c = capsule()
    c["operation"]["attributes"]["epilogue"] = "relu"
    with pytest.raises(ValueError, match="sequence"):
        O.public_cohort([c])


@pytest.mark.parametrize("raw", [b'{"schema":"first","schema":"second"}', b'{"facts":NaN}', b'{"facts":1e999}'])
def test_coverage_request_refuses_ambiguous_metadata(tmp_path, raw):
    from merlin.targetgen.opset_coverage_input import read_request

    path = tmp_path / "request.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="duplicate JSON key|non-finite JSON value"):
        read_request(path)
