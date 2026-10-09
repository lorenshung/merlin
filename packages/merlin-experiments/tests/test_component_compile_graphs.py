"""Actual multi-output scale sources, source-bound owners and bounded topology."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from merlin_experiments.phase0 import component_compile_graphs as G
from merlin_experiments.phase0 import component_compile_plan as P
from merlin_experiments.phase0 import component_compile_sources as S
from merlin_experiments.phase0 import component_coverage, software_intake
from merlin_experiments.phase0 import component_graph_variants as F
from merlin_experiments.phase0.rtl_intake import RtlIntakeRefusal
from test_component_compile_sources import (  # noqa: F401 -- registered independent native source fixtures
    _write,
    contraction_selection,
    literal,
    no_data,
    selected,
    selection,
)

from merlin.targetgen.contract.linalg_iface import parse_linalg_mlir


def add_selection(options):
    contraction_selection(options)
    review = json.loads(options["review"].read_bytes())
    path = Path(review["semantic_basis"]["path"])
    basis = json.loads(path.read_bytes())
    graph = {
        "schema": "m2m.frontend_graph.v1",
        "stage": "original",
        "status": "complete",
        "call_count": 1,
        "by_target": {"aten.add.Tensor": 1},
        "nodes": [
            {
                "id": name,
                "op": "placeholder",
                "target": name,
                "results": [{"id": name, "shape": [2, 4], "dtype": "i32"}],
            }
            for name in ("x", "y")
        ]
        + [
            {
                "id": "sum",
                "op": "call_function",
                "target": "aten.add.Tensor",
                "results": [{"id": "result", "shape": [2, 4], "dtype": "i32"}],
            }
        ],
        "edges": [
            {
                "producer_node_id": name,
                "consumer_node_id": "sum",
                "producer_value_id": name,
                "shape": [2, 4],
                "dtype": "i32",
            }
            for name in ("x", "y")
        ],
    }
    graph["sha256"] = component_coverage.digest(graph)
    pin = _write(
        options["source"].with_name("independent-add.json"),
        {"schema": "m2m.frontend_trace.v1", "graphs": {"original": graph}},
    )
    basis["members"].append(
        {
            "id": "independent-add",
            "kind": "model2mlir_frontend_trace",
            **pin,
            "schema": "m2m.frontend_trace.v1",
            "operation_semantics": ["aten.add.Tensor"],
            "effect_semantics": [],
        }
    )
    review["semantic_basis"] = _write(path, basis)
    source = json.loads(options["source"].read_bytes())
    source["operations"]["elementwise"] = {"families": ["elementwise_map"], "hardware": "standalone"}
    review["source"] = _write(options["source"], source)
    review["operation_basis"].append(
        {"owner": "elementwise", "member": "independent-add", "operations": ["aten.add.Tensor"]}
    )
    _write(options["review"], review)


def options_for(options, tmp_path):
    software = software_intake.issue_independent_software_intake(**options)
    return {
        "hardware": software.hardware,
        "software": software,
        "target_descriptor": Path(
            next(pin.path for pin in software.hardware.source_pins if pin.role == "target-descriptor")
        ),
        "plan": tmp_path / "source-graph-plan.json",
        "forbidden_roots": options["forbidden_roots"],
        "output_root": tmp_path / "source-graph-roster",
    }


@pytest.fixture
def configured(selection, tmp_path):  # noqa: F811 -- registered actual source selection fixture
    add_selection(selection)
    return options_for(selection, tmp_path)


def graph_member(name="graph-guard", cohort="functional_guard", *, depth=2, fanout=3, extent=10**9):
    return {
        "name": name,
        "cohort": cohort,
        "expectation": "compile_only",
        "operation": "component_program",
        "operation_owners": {"matmul": "contraction", "copy": "movement", "add": "elementwise"},
        "dimensions": {"M": literal(extent), "K": literal(3), "N": literal(5)},
        "depth": depth,
        "fanout": fanout,
        "graph_family": {"schema": F.SCHEMA, "representations": list(F.VARIANTS)},
        "required_static_obligations": list(P.OBLIGATIONS) + ["dependency_legality", "input_numeric_domain"],
    }


def select(options, rows, **budget):
    result = {
        "schema": G.SCHEMA,
        "provenance": dict(P.PROVENANCE),
        "hardware_intake_sha256": options["hardware"].sha256,
        "software_intake_sha256": options["software"].sha256,
        "budget": {
            "max_members": 8,
            "max_source_bytes": 200000,
            "max_extent_bits": 64,
            "max_scalar_bits": 64,
            "max_nodes": 500,
            "max_outputs": 100,
        },
        "members": rows,
    }
    result["budget"].update(budget)
    _write(options["plan"], result)
    return result


def test_large_graph_pair_reopens_every_original_snapshot_escape_and_final_output(configured, monkeypatch):
    no_data(monkeypatch)
    select(configured, [graph_member()])
    roster = S.issue_independent_compile_only_roster(**configured)
    roster.verify()
    assert [member.name for member in roster.members] == ["graph-guard__logical_epochs", "graph-guard__fresh_values"]
    assert len({member.source_sha256 for member in roster.members}) == 2
    expected = [f"E{stage}_{kind}" for stage in range(2) for kind in ("old", "escaped", "published")] + ["Final"]
    for member in roster.members:
        assert [slot.name for slot in member.original_abi.outputs] == expected
        assert all(slot.shape == (10**9, 5) and slot.dtype == "i32" for slot in member.original_abi.outputs)
        assert [(slot.name, slot.shape, slot.dtype) for slot in member.original_abi.inputs] == [
            ("A", (10**9, 3), "i8"),
            ("W", (3, 5), "i8"),
        ]
        source = member.source.read_text()
        parsed = parse_linalg_mlir(source)
        assert len(parsed["results"]) == 7 and len(parsed["ops"]) > len(parsed["results"])
        assert "dense<" not in source and source.count("func.return") == 1
        assert parsed["ops"][0]["body_ops"] == ["arith.extsi", "arith.extsi", "arith.muli", "arith.addi"]
        # Multiple independent copies and the later arithmetic read the original
        # producer. This is a real parsed source fork, not just a roster tag.
        first = [operand for op in parsed["ops"] for operand in op["ins"] if operand["source"] == ("op", 0)]
        assert len(first) > 2
    report = json.loads(roster.receipt_json)
    assert report["required_members"] == report["source_ready"] == 2
    assert report["source_metadata"]["outputs"] == 14
    graphs = [row["derivation"]["graph"] for row in report["members"]]
    assert graphs[0]["logical_epochs"] == {"P": 2} and graphs[1]["logical_epochs"] == {}
    assert "alias" in graphs[0]["effects"] and "shared_producer" in graphs[1]["effects"]
    assert all(set(row["static_status"].values()) == {"unknown"} for row in report["members"])
    assert all("physical reuse" in graph["scope"] for graph in graphs)
    assert not list(configured["output_root"].rglob("*.npy"))


def test_declared_depth_and_fanout_change_actual_ir_and_complete_output_roster(configured, monkeypatch):
    no_data(monkeypatch)
    select(
        configured,
        [graph_member(depth=1, fanout=2), graph_member("private-graph", "withheld_transfer", depth=3, fanout=4)],
    )
    roster = S.issue_independent_compile_only_roster(**configured)
    assert len(roster.members) == 4
    guard, transfer = roster.members[0], roster.members[2]
    assert len(guard.original_abi.outputs) == 4 and len(transfer.original_abi.outputs) == 10
    assert len(parse_linalg_mlir(transfer.source.read_text())["ops"]) > len(
        parse_linalg_mlir(guard.source.read_text())["ops"]
    )
    public = json.dumps(roster.public_summary())
    assert "private-graph" not in public and "fanout" not in public and "extents" not in public
    assert roster.public_summary()["cohorts"] == {"functional_guard": 2, "withheld_transfer": 2}


def test_two_owner_minimal_spec_does_not_invent_elementwise_source_support(selection, tmp_path, monkeypatch):  # noqa: F811
    contraction_selection(selection)
    options = options_for(selection, tmp_path)
    no_data(monkeypatch)
    select(options, [graph_member()])
    original = F.program

    def bounded(values, variant):
        assert values["depth"] <= 1
        return original(values, variant)

    monkeypatch.setattr(F, "program", bounded)
    with pytest.raises(S.CompileOnlyRosterRefusal) as result:
        S.issue_independent_compile_only_roster(**options)
    report = json.loads(result.value.report_path.read_bytes())
    assert report["required_members"] == 2 and report["source_ready"] == 0
    assert all(
        "operation add has no selected independent semantic owner" in row["missing"][0] for row in report["members"]
    )
    assert all(set(row["static_status"].values()) == {"unknown"} for row in report["members"])


def test_graph_copy_signature_must_admit_actual_accumulator_producers(selection, tmp_path, monkeypatch):  # noqa: F811
    add_selection(selection)
    source = json.loads(selection["source"].read_bytes())
    source["operations"]["movement"]["signature"] = {"ordered_operand_dtypes": ["int8"]}
    review = json.loads(selection["review"].read_bytes())
    review["source"] = _write(selection["source"], source)
    _write(selection["review"], review)
    options = options_for(selection, tmp_path)
    select(options, [graph_member()])
    no_data(monkeypatch)
    with pytest.raises(S.CompileOnlyRosterRefusal) as result:
        S.issue_independent_compile_only_roster(**options)
    report = json.loads(result.value.report_path.read_bytes())
    assert report["required_members"] == 2 and report["source_ready"] == 0
    assert all("ordered_operand_dtypes" in row["missing"][0] for row in report["members"])


@pytest.mark.parametrize("parameter", ["depth", "fanout"])
def test_huge_graph_is_denied_before_unroll_data_or_source_render(configured, monkeypatch, parameter):
    no_data(monkeypatch)
    row = graph_member()
    row[parameter] = 10**9
    select(configured, [row])
    original = F.program
    seen = []

    def bounded(values, variant):
        assert values["depth"] <= 1 and values["fanout"] <= 3
        seen.append((values["depth"], values["fanout"]))
        return original(values, variant)

    monkeypatch.setattr(F, "program", bounded)
    monkeypatch.setattr(
        P.component_program, "render", lambda *args, **kwargs: pytest.fail("over-budget topology reached renderer")
    )
    with pytest.raises(S.CompileOnlyRosterRefusal) as result:
        S.issue_independent_compile_only_roster(**configured)
    report = json.loads(result.value.report_path.read_bytes())
    assert report["required_members"] == 2 and report["source_ready"] == 0
    assert all("before topology construction" in row["missing"][0] for row in report["members"])
    assert seen and max(depth for depth, _ in seen) == 1


def test_pair_budget_counts_both_representations_and_prevents_second_full_unroll(configured, monkeypatch):
    no_data(monkeypatch)
    row = graph_member(depth=2, fanout=2)
    select(configured, [row], max_nodes=20)
    original, full = F.program, []

    def observe(values, variant):
        if values["depth"] == row["depth"]:
            full.append(variant)
        return original(values, variant)

    monkeypatch.setattr(F, "program", observe)
    with pytest.raises(S.CompileOnlyRosterRefusal) as result:
        S.issue_independent_compile_only_roster(**configured)
    report = json.loads(result.value.report_path.read_bytes())
    assert report["required_members"] == 2 and report["source_ready"] == 1
    assert full == ["logical_epochs"]
    assert "complete nodes budget before topology construction" in report["members"][1]["missing"][0]


def test_output_metadata_budget_cannot_drop_escaped_intermediates(configured, monkeypatch):
    no_data(monkeypatch)
    select(configured, [graph_member(depth=3)], max_outputs=9)
    with pytest.raises(S.CompileOnlyRosterRefusal) as result:
        S.issue_independent_compile_only_roster(**configured)
    report = json.loads(result.value.report_path.read_bytes())
    assert report["required_members"] == 2 and report["source_ready"] == 0
    assert all("outputs budget exceeded before topology construction" in row["missing"][0] for row in report["members"])


def test_missing_representation_or_graph_obligation_refuses_protected_plan(configured):
    row = graph_member()
    row["graph_family"]["representations"].pop()
    select(configured, [row])
    with pytest.raises(ValueError, match="complete representation pair"):
        S.issue_independent_compile_only_roster(**configured)
    assert not configured["output_root"].exists()
    row = graph_member()
    row["required_static_obligations"].remove("dependency_legality")
    select(configured, [row])
    with pytest.raises(RtlIntakeRefusal, match="dependency"):
        S.issue_independent_compile_only_roster(**configured)


def test_original_multioutput_abi_or_source_pair_cannot_be_replaced_by_final_only(configured):
    select(configured, [graph_member()])
    roster = S.issue_independent_compile_only_roster(**configured)
    member = roster.members[0]
    shortened = replace(member, original_abi=replace(member.original_abi, outputs=(member.original_abi.outputs[-1],)))
    object.__setattr__(roster, "members", (shortened, roster.members[1]))
    with pytest.raises(RtlIntakeRefusal, match="live independently issued"):
        roster.verify()


def test_expanded_names_cannot_collide_with_an_original_primitive_member(configured):
    from test_component_compile_sources import member

    select(configured, [graph_member(), member("graph-guard__logical_epochs")])
    with pytest.raises(RtlIntakeRefusal, match="staging name"):
        S.issue_independent_compile_only_roster(**configured)
    assert not configured["output_root"].exists()
