"""Neutral private joins for reviewed, closed integer-reduction source bodies."""

# ruff: noqa: E501 - emitted MLIR lines retain the producer's operation form.

from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256

import pytest
from merlin_experiments.phase1.feedback import private_integer_reduction_support as support

from merlin.common import mlir_query as mq
from merlin.frontends.capture_normalization import normalize_capture_mlir


def _selected() -> dict:
    return {
        "schema": "merlin.selected-index-lowering.v1",
        "compiler_requested": "neutral-clang",
        "compiler_resolved": "/neutral/clang",
        "compiler_sha256": "a" * 64,
        "cross_flags": ["--target=neutral"],
        "data_layout": "e-p:64:64",
        "index_bits": 64,
        "scope": "neutral test premise",
    }


def _actual(selected: dict) -> dict:
    return {
        **selected,
        "effective_pipeline": ",".join(
            f"{name}{{index-bitwidth=64}}"
            for name in (
                "convert-index-to-llvm",
                "convert-arith-to-llvm",
                "finalize-memref-to-llvm",
                "convert-func-to-llvm",
                "convert-cf-to-llvm",
            )
        ),
    }


def _source(kind: str) -> str:
    tag = {"sum": "aten.sum.dim_IntList", "cumsum": "aten.cumsum.default", "min": "aten.min.dim"}[kind]
    if kind == "sum":
        return f'''builtin.module {{
  func.func @forward(%arg0: tensor<2x3xi64>) -> tensor<2xi64> {{
    %zero = "arith.constant"() <{{value = 0 : i64}}> : () -> i64
    %init = "tensor.splat"(%zero) : (i64) -> tensor<2xi64>
    %sum = "linalg.reduce"(%arg0, %init) <{{dimensions = array<i64: 1>}}> ({{
    ^bb1(%elem: i64, %acc: i64):
      %added = "arith.addi"(%elem, %acc) <{{overflowFlags = #arith.overflow<none>}}> : (i64, i64) -> i64
      "linalg.yield"(%added) : (i64) -> ()
    }}) {{prov.aten = "{tag}"}} : (tensor<2x3xi64>, tensor<2xi64>) -> tensor<2xi64>
    func.return %sum : tensor<2xi64>
  }}
}}'''
    if kind == "cumsum":
        return f'''builtin.module {{
  func.func @forward(%arg0: tensor<2x3xi64>) -> tensor<2x3xi64> {{
    %zero = "arith.constant"() <{{value = 0 : i64}}> : () -> i64
    %init = "tensor.splat"(%zero) : (i64) -> tensor<2x3xi64>
    %scan = "linalg.generic"(%arg0, %init) <{{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 1, 1>}}> ({{
    ^bb0(%value: i64, %acc: i64):
      %i = "linalg.index"() <{{dim = 1 : i64}}> : () -> index
      %j = "linalg.index"() <{{dim = 2 : i64}}> : () -> index
      %mask = "arith.cmpi"(%j, %i) <{{predicate = 7 : i64}}> : (index, index) -> i1
      %selected = "arith.select"(%mask, %value, %zero) : (i1, i64, i64) -> i64
      %added = "arith.addi"(%acc, %selected) <{{overflowFlags = #arith.overflow<none>}}> : (i64, i64) -> i64
      "linalg.yield"(%added) : (i64) -> ()
    }}) {{prov.aten = "{tag}"}} : (tensor<2x3xi64>, tensor<2x3xi64>) -> tensor<2x3xi64>
    func.return %scan : tensor<2x3xi64>
  }}
}}'''
    return f'''builtin.module {{
  func.func @pair(%x: tensor<2x5xi64>) -> (tensor<2xi64>, tensor<2xi64>) {{
    %seed = arith.constant 9223372036854775807 : i64
    %vinit = tensor.splat %seed : tensor<2xi64>
    %iinit = tensor.splat %seed : tensor<2xi64>
    %value, %index = "linalg.generic"(%x, %vinit, %iinit) <{{
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      operandSegmentSizes = array<i32: 1, 2>}}> ({{
      ^bb0(%input: i64, %old: i64, %old_index: i64):
        %coordinate = "linalg.index"() <{{dim = 1 : i64}}> : () -> index
        %position = arith.index_cast %coordinate : index to i64
        %better = arith.cmpi slt, %input, %old : i64
        %same = arith.cmpi eq, %input, %old : i64
        %earlier = arith.cmpi ult, %position, %old_index : i64
        %tie = arith.andi %same, %earlier : i1
        %choose = arith.ori %better, %tie : i1
        %new_value = arith.select %choose, %input, %old : i64
        %new_index = arith.select %choose, %position, %old_index : i64
        "linalg.yield"(%new_value, %new_index) : (i64, i64) -> ()
    }}) {{prov.aten = "{tag}"}} : (tensor<2x5xi64>, tensor<2xi64>, tensor<2xi64>) -> (tensor<2xi64>, tensor<2xi64>)
    func.return %value, %index : tensor<2xi64>, tensor<2xi64>
  }}
}}'''


def _boolean_source(kind: str) -> str:
    text = _source(kind)
    assert kind in {"sum", "cumsum"}
    if kind == "sum":
        cast = """    %empty = "tensor.empty"() : () -> tensor<2x3xi64>
    %cast = "linalg.generic"(%arg0, %empty) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%bit: i1, %unused: i64):
      %word = "arith.extui"(%bit) : (i1) -> i64
      "linalg.yield"(%word) : (i64) -> ()
    }) : (tensor<2x3xi1>, tensor<2x3xi64>) -> tensor<2x3xi64>
"""
        return (
            text.replace("%arg0: tensor<2x3xi64>", "%arg0: tensor<2x3xi1>")
            .replace('    %zero = "arith.constant"', cast + '    %zero = "arith.constant"')
            .replace('"linalg.reduce"(%arg0, %init)', '"linalg.reduce"(%cast, %init)')
        )
    return (
        text.replace("%arg0: tensor<2x3xi64>", "%arg0: tensor<2x3xi1>")
        .replace("^bb0(%value: i64, %acc: i64):", "^bb0(%value: i1, %acc: i64):")
        .replace(
            '      %selected = "arith.select"(%mask, %value, %zero)',
            '      %word = "arith.extui"(%value) : (i1) -> i64\n      %selected = "arith.select"(%mask, %word, %zero)',
        )
        .replace('"linalg.generic"(%arg0, %init)', '"linalg.generic"(%arg0, %init)')
        .replace(
            ": (tensor<2x3xi64>, tensor<2x3xi64>) -> tensor<2x3xi64>",
            ": (tensor<2x3xi1>, tensor<2x3xi64>) -> tensor<2x3xi64>",
        )
    )


def _prepared(tmp_path, kind="sum", *, boolean=False):
    path = tmp_path / "neutral.mlir"
    raw = _boolean_source(kind) if boolean else _source(kind)
    path.write_text(raw)
    normalized, receipt = normalize_capture_mlir(raw)
    parsed = tuple(mq.walk(mq.parse(normalized)))
    op_name = "linalg.reduce" if kind == "sum" else "linalg.generic"
    ordinals = [i for i, op in enumerate(parsed) if mq.op_name(op) == op_name]
    assert len(ordinals) == 1
    tag = {"sum": "aten.sum.dim_IntList", "cumsum": "aten.cumsum.default", "min": "aten.min.dim"}[kind]
    row = {"mlir_operation": op_name, "frontend_op": tag, "ordinals": ordinals, "count": 1}
    selected = _selected()
    witness = support.begin(sha256(raw.encode()).hexdigest(), receipt["output_sha256"], len(parsed), selected)
    source = {
        "source_sha256": witness["raw_source_sha256"],
        "normalized_source_sha256": witness["normalized_source_sha256"],
        "n_source_operations": len(parsed),
        "selected_index_observation": selected,
        support.FIELD: witness,
    }
    decision = {
        "status": "admitted",
        "reviewed": True,
        "profiles": [
            {"status": "admitted", "reviewed": True, "profile": "neutral-reviewed", "capability_spec_sha256": "b" * 64}
        ],
    }
    return path, parsed, row, source, decision


def _linked(source, selected):
    triple = {"candidate_tree_sha256": "c" * 64, "capture_tree_sha256": "d" * 64, "elf_sha256": "e" * 64}
    actual = _actual(selected)
    support.link(source, actual, triple)
    entry = {**triple, "source_sha256": source["source_sha256"], "index_lowering": actual}
    return entry


@pytest.mark.parametrize("kind", ["sum", "cumsum", "min"])
def test_reviewed_source_and_exact_linked_build(kind, tmp_path):
    path, parsed, row, source, decision = _prepared(tmp_path, kind)
    support.record(source[support.FIELD], row, decision, parsed, {row["ordinals"][0]: row})
    support.verify_source(source[support.FIELD], path)
    entry = _linked(source, source["selected_index_observation"])
    assert support.linked_complete(source, entry, "c" * 64)
    assert support.linked_complete(json.loads(json.dumps(source)), json.loads(json.dumps(entry)), "c" * 64)
    assert source[support.FIELD]["count"] == 1
    assert source[support.FIELD]["occurrences"][0]["kind"] == ("i64_min_first_index" if kind == "min" else kind)
    for field in ("capture_tree_sha256", "elf_sha256", "index_lowering"):
        damaged = deepcopy(entry)
        damaged.pop(field)
        assert not support.linked_complete(source, damaged, "c" * 64)
    assert not support.linked_complete(source, entry, "f" * 64)


@pytest.mark.parametrize("kind", ["sum", "cumsum", "min"])
def test_host_screen_routes_exact_reduction_proof_to_mandatory_integer_witness(kind, tmp_path):
    from merlin_experiments.phase1.feedback import private_full_models as full
    from merlin_experiments.phase1.feedback import private_linalg_support as linalg

    from merlin.targetgen.application_inventory import operation_structure
    from merlin.targetgen.host_capabilities import admit_host_operation

    _path, parsed, row, source, _decision = _prepared(tmp_path, kind)
    ordinal = row["ordinals"][0]
    structure = operation_structure(parsed[ordinal])
    inputs, outputs = structure["ordered_operand_types"], structure["ordered_result_types"]
    row = {**row, **structure, "semantic_family": "reduction"}
    observed = {
        "family": "reduction",
        "ordered_operand_dtypes": [item["dtype"] for item in inputs],
        "ordered_result_dtypes": [item["dtype"] for item in outputs],
        "rank": len(outputs[0]["shape"]),
    }
    declaration = {
        "id": "neutral_integer_reduction",
        "ops": [row["frontend_op"]],
        "families": ["reduction"],
        "placement": "host",
        "signature": {
            "family": "reduction",
            "ordered_operand_dtypes": observed["ordered_operand_dtypes"],
            "ordered_result_dtypes": observed["ordered_result_dtypes"],
        },
        "source_body": {
            "schema": "merlin.static_integer_reduction_source_body.v1",
            "operation": "i64_min_first_index" if kind == "min" else kind,
        },
    }
    selected = {
        "neutral": {
            "package_sha256": "a" * 64,
            "capability_spec_sha256": "b" * 64,
            "dtype_strategy": "int8_w8a8",
            "capability_spec": {
                "schema": "merlin.host_capabilities.v1",
                "status": "reviewed",
                "compiler": {"package_sha256": "a" * 64, "dtype_strategy": "int8_w8a8"},
                "operations": [declaration],
                "evidence": {"scope": "neutral structural test, not numerical qualification"},
            },
        }
    }
    decision = admit_host_operation(
        selected,
        row,
        observed,
        source_operations=(parsed[ordinal],),
        source_context={"selected_index_observation": source["selected_index_observation"]},
    )
    assert decision["status"] == "admitted"

    def record(proof):
        integer = support.begin(
            source["source_sha256"],
            source["normalized_source_sha256"],
            len(parsed),
            source["selected_index_observation"],
        )
        pointwise = linalg.begin(
            source["source_sha256"],
            source["normalized_source_sha256"],
            len(parsed),
            source["selected_index_observation"],
        )
        full._record_linalg_or_integer_source(pointwise, integer, row, proof, parsed, {ordinal: row}, None)
        return pointwise, integer

    pointwise, integer = record(decision)
    assert pointwise["count"] == 0 and integer["count"] == 1
    assert integer["occurrences"][0]["ordinal"] == ordinal
    assert record(json.loads(json.dumps(decision)))[1]["count"] == 1
    for edit in (
        lambda bad: bad["source_body_proof"]["patterns"][0].update(axis=(99,)),
        lambda bad: bad["source_body_proof"]["selected_index_observation"].update(index_bits=32),
        lambda bad: bad["source_body_proof"]["selected_index_observation"].update(index_bits=64.0),
        lambda bad: bad["source_body_proof"]["patterns"][0].update(axis=[True] if kind != "min" else True),
        lambda bad: bad["source_body_proof"]["patterns"][0].update(input_shape=[2.0, 3 if kind != "min" else 5]),
        lambda bad: bad["source_body_proof"]["patterns"][0].update(index_bits_premise=64.0)
        if kind != "min"
        else bad["source_body_proof"]["patterns"][0].update(index_bits=64.0),
        lambda bad: bad["profiles"][0]["decisions"][0]["source_body_proof"]["patterns"][0].update(
            input_shape=[2.0, 3 if kind != "min" else 5]
        ),
        lambda bad: bad["profiles"][0]["decisions"][0]["source_body_proof"][
            "selected_index_observation"
        ].update(index_bits=64.0),
        lambda bad: bad["source_body_proof"].update(declaration="forged"),
        lambda bad: bad.pop("source_body_proof"),
    ):
        bad = deepcopy(decision)
        edit(bad)
        with pytest.raises(ValueError, match="source-body admission|parsed ordinal|reviewed declaration"):
            record(bad)


def test_reviewed_admission_and_exact_source_are_mandatory(tmp_path):
    path, parsed, row, source, decision = _prepared(tmp_path)
    bad = deepcopy(decision)
    bad["reviewed"] = False
    with pytest.raises(ValueError, match="reviewed host admission"):
        support.record(source[support.FIELD], row, bad, parsed, {row["ordinals"][0]: row})
    support.record(source[support.FIELD], row, decision, parsed, {row["ordinals"][0]: row})
    with pytest.raises(ValueError, match="repeated"):
        support.record(source[support.FIELD], row, decision, parsed, {row["ordinals"][0]: row})
    path.write_text(path.read_text().replace("value = 0 : i64", "value = 1 : i64"))
    with pytest.raises(ValueError):
        support.verify_source(source[support.FIELD], path)
    with pytest.raises(ValueError, match="source witness is not pending|source premise"):
        support.link(source, _actual(_selected()), {"candidate_tree_sha256": "c" * 64})


@pytest.mark.parametrize("kind", ["sum", "cumsum"])
def test_boolean_input_cast_is_proved_without_claiming_direct_boolean_abi(kind, tmp_path):
    path, parsed, row, source, decision = _prepared(tmp_path, kind, boolean=True)
    support.record(source[support.FIELD], row, decision, parsed, {row["ordinals"][0]: row})
    support.verify_source(source[support.FIELD], path)
    entry = _linked(source, source["selected_index_observation"])
    assert source[support.FIELD]["occurrences"][0]["pattern"]["input_type"] == "i1"
    assert support.linked_complete(source, entry, "c" * 64)


def test_serialized_witness_rejects_tampered_roster_and_width(tmp_path):
    path, parsed, row, source, decision = _prepared(tmp_path, "min")
    support.record(source[support.FIELD], row, decision, parsed, {row["ordinals"][0]: row})
    support.verify_source(source[support.FIELD], path)
    entry = _linked(source, source["selected_index_observation"])
    for field, replacement in (("axis", 0), ("ordered_types", ["i64"] * 4), ("index_bits", 32)):
        damaged = deepcopy(source)
        damaged[support.FIELD]["occurrences"][0]["pattern"][field] = replacement
        damaged[support.FIELD]["occurrences_sha256"] = support._digest(damaged[support.FIELD]["occurrences"])
        assert not support.linked_complete(damaged, entry, "c" * 64)
    damaged = deepcopy(source)
    damaged[support.FIELD]["occurrences"][0]["ordinal"] = True
    damaged[support.FIELD]["occurrences_sha256"] = support._digest(damaged[support.FIELD]["occurrences"])
    assert not support.linked_complete(damaged, entry, "c" * 64)


def test_selected_build_and_reviewed_profile_refuse_missing_or_changed_identity(tmp_path):
    path, parsed, row, source, decision = _prepared(tmp_path, "cumsum")
    bad = deepcopy(decision)
    bad["profiles"][0]["capability_spec_sha256"] = "wrong"
    with pytest.raises(ValueError, match="unique reviewed host profile"):
        support.record(source[support.FIELD], row, bad, parsed, {row["ordinals"][0]: row})
    wrong_row = {**row, "mlir_operation": "linalg.reduce"}
    with pytest.raises(ValueError, match="parsed operation"):
        support.record(source[support.FIELD], wrong_row, decision, parsed, {row["ordinals"][0]: wrong_row})
    wrong_frontend = {**row, "frontend_op": "aten.add.Tensor"}
    with pytest.raises(ValueError, match="matching frontend identity"):
        support.record(source[support.FIELD], wrong_frontend, decision, parsed, {row["ordinals"][0]: wrong_frontend})
    support.record(source[support.FIELD], row, decision, parsed, {row["ordinals"][0]: row})
    support.verify_source(source[support.FIELD], path)
    selected = source["selected_index_observation"]
    changed_index = _actual(selected)
    changed_index["index_bits"] = 32
    with pytest.raises(ValueError, match="actual linked compiler index"):
        support.link(
            source,
            changed_index,
            {"candidate_tree_sha256": "c" * 64, "capture_tree_sha256": "d" * 64, "elf_sha256": "e" * 64},
        )
    with pytest.raises(ValueError, match="linked candidate/capture/ELF"):
        support.link(source, _actual(selected), {"candidate_tree_sha256": "c" * 64})
    entry = _linked(source, selected)
    damaged = deepcopy(entry)
    damaged["index_lowering"]["effective_pipeline"] = "convert-index-to-llvm{index-bitwidth=32}"
    assert not support.linked_complete(source, damaged, "c" * 64)


def test_unrelated_source_row_does_not_select_integer_reduction_obligation(tmp_path):
    _, parsed, _, source, _ = _prepared(tmp_path)
    ordinal = next(i for i, op in enumerate(parsed) if mq.op_name(op) == "tensor.splat")
    row = {"mlir_operation": "tensor.splat", "frontend_op": None, "ordinals": [ordinal], "count": 1}
    support.record(source[support.FIELD], row, {"status": "unsupported"}, parsed, {ordinal: row})
    assert source[support.FIELD]["count"] == 0


def test_missing_selected_width_only_allows_empty_source_diagnostic(tmp_path):
    path, parsed, row, source, decision = _prepared(tmp_path)
    witness = support.begin(source["source_sha256"], source["normalized_source_sha256"], len(parsed), None)
    support.verify_source(witness, path)
    assert witness["count"] == 0
    with pytest.raises(ValueError, match="selected producer-owned index width"):
        support.record(witness, row, decision, parsed, {row["ordinals"][0]: row})
    source[support.FIELD] = witness
    source["selected_index_observation"] = None
    with pytest.raises(ValueError, match="actual linked compiler index"):
        support.link(
            source,
            _actual(_selected()),
            {"candidate_tree_sha256": "c" * 64, "capture_tree_sha256": "d" * 64, "elf_sha256": "e" * 64},
        )
    assert not support.linked_complete(
        source,
        {"source_sha256": source["source_sha256"], "index_lowering": _actual(_selected())},
        "c" * 64,
    )
