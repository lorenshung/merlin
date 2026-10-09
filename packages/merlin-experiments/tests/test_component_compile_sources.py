"""Live source-only roster construction, complete denials and original ABI replay."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from merlin_experiments.phase0 import component_compile_plan as P
from merlin_experiments.phase0 import component_compile_sources as S
from merlin_experiments.phase0 import component_coverage, component_numerics, software_intake
from merlin_experiments.phase0.rtl_intake import RtlIntakeRefusal
from test_independent_software_intake import (  # noqa: F401 -- actual independent native RTL/SW replay
    selected,
    selection,
)

from merlin.targetgen import capsule_inputs, component_program, golden_store
from merlin.targetgen.contract.linalg_iface import parse_linalg_mlir


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def literal(value):
    return {"kind": "integer", "value": value}


def member(name="copy-guard", cohort="functional_guard", operation="copy", **dimensions):
    return {
        "name": name,
        "cohort": cohort,
        "expectation": "compile_only",
        "operation": operation,
        "operation_owner": "movement" if operation == "copy" else "contraction",
        "dimensions": dimensions or {"M": literal(2), "N": literal(3)},
        "required_static_obligations": list(P.OBLIGATIONS) + ["input_numeric_domain"],
    }


@pytest.fixture
def configured(selection, tmp_path):  # noqa: F811 -- actual source qualification fixture
    software = software_intake.issue_independent_software_intake(**selection)
    hardware = software.hardware
    descriptor = Path(next(pin.path for pin in hardware.source_pins if pin.role == "target-descriptor"))
    path = tmp_path / "source-plan.json"
    return {
        "hardware": hardware,
        "software": software,
        "target_descriptor": descriptor,
        "plan": path,
        "forbidden_roots": selection["forbidden_roots"],
        "output_root": tmp_path / "source-roster",
    }


def plan(options, rows, **budget):
    value = {
        "schema": P.SCHEMA,
        "provenance": dict(P.PROVENANCE),
        "hardware_intake_sha256": options["hardware"].sha256,
        "software_intake_sha256": options["software"].sha256,
        "budget": {"max_members": 5, "max_source_bytes": 10000, "max_extent_bits": 64, "max_scalar_bits": 64},
        "members": rows,
    }
    value["budget"].update(budget)
    _write(options["plan"], value)
    return value


def no_data(monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("source-only generation reached operand, golden, numerical or capsule binding construction")

    monkeypatch.setattr(capsule_inputs, "materialize_capsule_leaves", refuse)
    monkeypatch.setattr(component_numerics, "evaluate", refuse)
    monkeypatch.setattr(golden_store, "load_golden", refuse)
    monkeypatch.setattr(golden_store, "write_golden", refuse)
    monkeypatch.setattr(component_program, "build", refuse)


def test_actual_memory_boundary_sources_complete_original_abi_and_private_denominator(configured, monkeypatch):
    no_data(monkeypatch)
    boundary = {"kind": "memory_volume", "memory": 0, "dtype_role": "operand", "fixed_elements": 1}
    plan(
        configured,
        [
            member(M={**boundary, "offset": -1}, N=literal(1)),
            member("copy-transfer", "withheld_transfer", M={**boundary, "offset": 1}, N=literal(1)),
        ],
    )
    roster = S.issue_independent_compile_only_roster(**configured)
    roster.verify()
    assert type(roster.members) is tuple and roster.hardware is configured["hardware"]
    assert roster.software is configured["software"]
    assert [item.original_abi.inputs[0].shape for item in roster.members] == [(3, 1), (5, 1)]
    assert all(item.original_abi.outputs[0].shape == item.original_abi.inputs[0].shape for item in roster.members)
    assert roster.members[0].source_sha256 != roster.members[1].source_sha256
    report = json.loads(roster.receipt_json)
    assert report["required_members"] == report["source_ready"] == 2
    assert all(set(row["static_status"].values()) == {"unknown"} for row in report["members"])
    assert all(
        not row["derivation"]["tensor_values_allocated"] and not row["derivation"]["goldens_allocated"]
        for row in report["members"]
    )
    extent = report["members"][0]["derivation"]["extents"]["M"]
    assert extent["fact"] == "memories.0.bytes" and extent["observed"] == 4
    assert "index legality unknown" in extent["scope"]
    public = json.dumps(roster.public_summary())
    assert "copy-transfer" not in public and str(roster.members[1].source) not in public and "extents" not in public
    assert sorted(path.name for path in configured["output_root"].iterdir()) == [
        "copy-guard",
        "copy-transfer",
        "roster.json",
    ]


def test_large_copy_source_has_constant_work_no_tensor_values_or_goldens(configured, monkeypatch):
    no_data(monkeypatch)
    plan(configured, [member(M=literal(10**9), N=literal(3))])
    roster = S.issue_independent_compile_only_roster(**configured)
    observed = parse_linalg_mlir(roster.members[0].source.read_text())
    assert observed["args"] == [{"index": 0, "shape": [10**9, 3], "dtype": "i8"}]
    assert observed["results"] == [{"shape": [10**9, 3], "dtype": "i8"}]
    assert roster.members[0].source.stat().st_size < 1000
    assert list(configured["output_root"].rglob("*.npy")) == []


def contraction_selection(options):
    """Add a distinct original independent public source, never a validation graph."""
    review = json.loads(options["review"].read_bytes())
    roster_path = Path(review["semantic_basis"]["path"])
    basis = json.loads(roster_path.read_bytes())
    graph = {
        "schema": "m2m.frontend_graph.v1",
        "stage": "original",
        "status": "complete",
        "call_count": 1,
        "by_target": {"aten.matmul.default": 1},
        "nodes": [
            {
                "id": "left",
                "op": "placeholder",
                "target": "a",
                "results": [{"id": "a", "shape": [2, 3], "dtype": "int8"}],
            },
            {
                "id": "right",
                "op": "placeholder",
                "target": "b",
                "results": [{"id": "b", "shape": [3, 4], "dtype": "int8"}],
            },
            {
                "id": "product",
                "op": "call_function",
                "target": "aten.matmul.default",
                "results": [{"id": "p", "shape": [2, 4], "dtype": "i32"}],
            },
        ],
        "edges": [
            {
                "producer_node_id": node,
                "consumer_node_id": "product",
                "producer_value_id": value,
                "shape": shape,
                "dtype": "int8",
            }
            for node, value, shape in [("left", "a", [2, 3]), ("right", "b", [3, 4])]
        ],
    }
    graph["sha256"] = component_coverage.digest(graph)
    pin = _write(
        options["source"].with_name("independent-contraction.json"),
        {"schema": "m2m.frontend_trace.v1", "graphs": {"original": graph}},
    )
    basis["members"].append(
        {
            "id": "independent-contraction",
            "kind": "model2mlir_frontend_trace",
            **pin,
            "schema": "m2m.frontend_trace.v1",
            "operation_semantics": ["aten.matmul.default"],
            "effect_semantics": [],
        }
    )
    review["semantic_basis"] = _write(roster_path, basis)
    source = json.loads(options["source"].read_bytes())
    source["operations"]["contraction"] = {"families": ["contraction"], "hardware": "standalone"}
    review["source"] = _write(options["source"], source)
    review["operation_basis"].append(
        {"owner": "contraction", "member": "independent-contraction", "operations": ["aten.matmul.default"]}
    )
    _write(options["review"], review)


def test_large_contraction_reopens_real_scalar_fill_and_ordered_source_signature(selection, tmp_path, monkeypatch):  # noqa: F811
    contraction_selection(selection)
    software = software_intake.issue_independent_software_intake(**selection)
    options = {
        "hardware": software.hardware,
        "software": software,
        "target_descriptor": Path(
            next(pin.path for pin in software.hardware.source_pins if pin.role == "target-descriptor")
        ),
        "plan": tmp_path / "contraction-plan.json",
        "forbidden_roots": selection["forbidden_roots"],
        "output_root": tmp_path / "contraction-roster",
    }
    no_data(monkeypatch)
    plan(options, [member(operation="matmul", M=literal(10**9), K=literal(3), N=literal(5))])
    roster = S.issue_independent_compile_only_roster(**options)
    item = roster.members[0]
    assert [(tensor.name, tensor.shape, tensor.dtype) for tensor in item.original_abi.inputs] == [
        ("A", (10**9, 3), "i8"),
        ("W", (3, 5), "i8"),
    ]
    assert item.original_abi.outputs[0].record() == {"name": "Y", "shape": [10**9, 5], "dtype": "i32"}
    source = item.source.read_text()
    assert "linalg.fill" in source and "dense<" not in source
    parsed = parse_linalg_mlir(source)
    assert parsed["ops"][0]["body_ops"] == ["arith.extsi", "arith.extsi", "arith.muli", "arith.addi"]
    assert parsed["ops"][0]["outs"][0]["source"] == ("init", "fill")
    assert set(json.loads(roster.receipt_json)["members"][0]["static_status"].values()) == {"unknown"}


@pytest.mark.parametrize(
    "defect", ["unknown-fact", "unknown-operation", "unknown-owner", "width-budget", "count-budget", "source-budget"]
)
def test_source_failures_keep_every_required_member_and_refuse_live_issuance(configured, monkeypatch, defect):
    no_data(monkeypatch)
    rows = [member(), member("required-transfer", "withheld_transfer")]
    budget = {}
    if defect == "unknown-fact":
        rows[1]["dimensions"]["M"] = {"kind": "memory_depth", "memory": 1, "offset": 0}
    elif defect == "unknown-operation":
        rows[1]["operation"] = "unknown_primitive"
    elif defect == "unknown-owner":
        rows[1]["operation_owner"] = "missing_owner"
    elif defect == "width-budget":
        rows[1]["dimensions"]["M"] = literal(10**9)
        budget["max_extent_bits"] = 8
    elif defect == "count-budget":
        budget["max_members"] = 1
    else:
        budget["max_source_bytes"] = 1
    plan(configured, rows, **budget)
    with pytest.raises(S.CompileOnlyRosterRefusal) as result:
        S.issue_independent_compile_only_roster(**configured)
    report = json.loads(result.value.report_path.read_bytes())
    assert report["required_members"] == len(report["members"]) == 2
    assert report["source_ready"] < 2
    assert report["members"][1]["name"] == "required-transfer"
    assert report["members"][1]["source_status"] == "source_unavailable"
    assert all(set(row["static_status"].values()) == {"unknown"} for row in report["members"])


def test_saved_roster_or_modified_original_abi_cannot_issue_live_grading_authority(configured):
    plan(configured, [member()])
    roster = S.issue_independent_compile_only_roster(**configured)
    with pytest.raises(RtlIntakeRefusal, match="live independently issued"):
        replace(roster).verify()
    forged = replace(
        roster.members[0],
        original_abi=replace(roster.members[0].original_abi, outputs=roster.members[0].original_abi.inputs),
    )
    object.__setattr__(roster, "members", (forged,))
    with pytest.raises(RtlIntakeRefusal, match="live independently issued"):
        roster.verify()


@pytest.mark.parametrize("source", ["plan", "source", "receipt"])
def test_actual_original_plan_source_and_receipt_bytes_are_reopened(configured, source):
    plan(configured, [member()])
    roster = S.issue_independent_compile_only_roster(**configured)
    path = {
        "plan": configured["plan"],
        "source": roster.members[0].source,
        "receipt": configured["output_root"] / "roster.json",
    }[source]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(RtlIntakeRefusal, match="source changed"):
        roster.verify()


def test_static_refusal_is_retained_as_separate_expectation_not_a_source_pass(configured):
    row = member(M={"kind": "memory_depth", "memory": 0, "offset": 1}, N=literal(1))
    row["expectation"] = "static_refusal"
    plan(configured, [row])
    roster = S.issue_independent_compile_only_roster(**configured)
    assert roster.members[0].expectation == "static_refusal"
    assert roster.members[0].original_abi.inputs[0].shape == (5, 1)
    assert json.loads(roster.receipt_json)["members"][0]["source_status"] == "source_ready"
    assert "candidate_compilation" in roster.public_summary()["unknowns"]


def test_plan_rejects_workload_or_target_schedule_metadata_before_source_construction(configured, monkeypatch):
    value = plan(configured, [member()])
    value["members"][0]["schedule"] = "historical policy"
    _write(configured["plan"], value)
    monkeypatch.setattr(P, "produce", lambda *args, **kwargs: pytest.fail("invalid plan reached producer"))
    with pytest.raises(RtlIntakeRefusal, match="history metadata"):
        S.issue_independent_compile_only_roster(**configured)
    assert not configured["output_root"].exists()


def test_exact_descriptor_and_same_live_hardware_are_required(configured, tmp_path):
    plan(configured, [member()])
    different = tmp_path / "another-descriptor.yaml"
    different.write_bytes(configured["target_descriptor"].read_bytes())
    with pytest.raises(RtlIntakeRefusal, match="original selected hardware"):
        S.issue_independent_compile_only_roster(**{**configured, "target_descriptor": different})
    with pytest.raises(RtlIntakeRefusal, match="identical live"):
        S.issue_independent_compile_only_roster(**{**configured, "hardware": replace(configured["hardware"])})


def test_complete_source_budget_counts_both_public_and_private_members(configured):
    rows = [member(), member("private-transfer", "withheld_transfer")]
    declaration = plan(configured, rows)
    source, _, _ = P.produce(
        rows[0], hardware=configured["hardware"], software=configured["software"], budget=declaration["budget"]
    )
    plan(configured, rows, max_source_bytes=len(source.encode()) + 1)
    with pytest.raises(S.CompileOnlyRosterRefusal) as result:
        S.issue_independent_compile_only_roster(**configured)
    report = json.loads(result.value.report_path.read_bytes())
    assert report["required_members"] == 2 and report["source_ready"] == 1
    assert report["members"][1]["missing"] == ["source-only mandatory roster exceeds the complete source byte budget"]


def test_semantic_dtype_constraint_is_checked_against_actual_original_signature(selection, tmp_path):  # noqa: F811
    raw = json.loads(selection["source"].read_bytes())
    raw["operations"]["movement"]["signature"] = {"ordered_result_dtypes": ["i32"]}
    review = json.loads(selection["review"].read_bytes())
    review["source"] = _write(selection["source"], raw)
    _write(selection["review"], review)
    software = software_intake.issue_independent_software_intake(**selection)
    options = {
        "hardware": software.hardware,
        "software": software,
        "target_descriptor": Path(
            next(pin.path for pin in software.hardware.source_pins if pin.role == "target-descriptor")
        ),
        "plan": tmp_path / "constraint-plan.json",
        "forbidden_roots": selection["forbidden_roots"],
        "output_root": tmp_path / "constraint-roster",
    }
    plan(options, [member()])
    with pytest.raises(S.CompileOnlyRosterRefusal) as result:
        S.issue_independent_compile_only_roster(**options)
    assert "ordered_result_dtypes" in json.loads(result.value.report_path.read_bytes())["members"][0]["missing"][0]


def test_protected_path_refuses_before_reading_even_an_existing_plan(configured, monkeypatch):
    plan(configured, [member()])
    monkeypatch.setattr(Path, "read_bytes", lambda *args: pytest.fail("protected plan was opened"))
    with pytest.raises(RtlIntakeRefusal, match="protected implementation"):
        S.issue_independent_compile_only_roster(**{**configured, "forbidden_roots": (configured["plan"].parent,)})


@pytest.mark.parametrize("selector", ["family-id", "explicit-other-op"])
def test_minimal_owner_id_support_never_bypasses_an_explicit_original_op_roster(selection, tmp_path, selector):  # noqa: F811
    raw = json.loads(selection["source"].read_bytes())
    raw["operations"]["movement"].pop("families")
    if selector == "explicit-other-op":
        raw["operations"]["movement"]["ops"] = ["transpose"]
    review = json.loads(selection["review"].read_bytes())
    review["source"] = _write(selection["source"], raw)
    _write(selection["review"], review)
    software = software_intake.issue_independent_software_intake(**selection)
    options = {
        "hardware": software.hardware,
        "software": software,
        "target_descriptor": Path(
            next(pin.path for pin in software.hardware.source_pins if pin.role == "target-descriptor")
        ),
        "plan": tmp_path / "minimal-id-plan.json",
        "forbidden_roots": selection["forbidden_roots"],
        "output_root": tmp_path / "minimal-id-roster",
    }
    plan(options, [member()])
    if selector == "family-id":
        roster = S.issue_independent_compile_only_roster(**options)
        assert roster.members[0].original_abi.outputs[0].shape == (2, 3)
    else:
        with pytest.raises(S.CompileOnlyRosterRefusal) as result:
            S.issue_independent_compile_only_roster(**options)
        assert "semantic owner" in json.loads(result.value.report_path.read_bytes())["members"][0]["missing"][0]
