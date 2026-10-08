"""Neutral typed mask-count guards: source proof cannot become host admission."""

from __future__ import annotations

from hashlib import sha256

import pytest
from merlin_experiments.phase1.feedback import private_control_support as control

from merlin.common import mlir_query as mq


def _source(extent: int = 4) -> str:
    return f""""builtin.module"() ({{
  "func.func"() <{{sym_name = "forward",
    function_type = (tensor<{extent}xi1>, tensor<{extent}xi64>) -> tensor<?xi64>}}> ({{
  ^bb0(%mask: tensor<{extent}xi1>, %data: tensor<{extent}xi64>):
    %init = "tensor.empty"() : () -> tensor<{extent}xi64>
    %bits = "linalg.generic"(%mask, %init) <{{
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = [#linalg.iterator_type<parallel>],
      operandSegmentSizes = array<i32: 1, 1>}}> ({{
      ^bb1(%bit: i1, %old: i64):
        %wide = "arith.extui"(%bit) : (i1) -> i64
        "linalg.yield"(%wide) : (i64) -> ()
    }}) : (tensor<{extent}xi1>, tensor<{extent}xi64>) -> tensor<{extent}xi64>
    %zero64 = "arith.constant"() <{{value = 0 : i64}}> : () -> i64
    %seed = "tensor.splat"(%zero64) : (i64) -> tensor<i64>
    %sum = "linalg.reduce"(%bits, %seed) <{{dimensions = array<i64: 0>}}> ({{
      ^bb2(%cell: i64, %acc: i64):
        %next = "arith.addi"(%cell, %acc) <{{overflowFlags = #arith.overflow<none>}}> : (i64, i64) -> i64
        "linalg.yield"(%next) : (i64) -> ()
    }}) : (tensor<{extent}xi64>, tensor<i64>) -> tensor<i64>
    %count = "tensor.extract"(%sum) : (tensor<i64>) -> i64
    %size = "arith.index_cast"(%count) : (i64) -> index
    %empty = "tensor.empty"(%size) : (index) -> tensor<?xi64>
    %lo = "arith.constant"() <{{value = 0 : index}}> : () -> index
    %step = "arith.constant"() <{{value = 1 : index}}> : () -> index
    %hi = "arith.constant"() <{{value = {extent} : index}}> : () -> index
    %result, %taken = "scf.for"(%lo, %hi, %step, %empty, %lo) ({{
      ^bb3(%i: index, %current: tensor<?xi64>, %index: index):
        %choose = "tensor.extract"(%mask, %i) : (tensor<{extent}xi1>, index) -> i1
        %new, %new_index = "scf.if"(%choose) ({{
          %item = "tensor.extract"(%data, %i) : (tensor<{extent}xi64>, index) -> i64
          %inserted = "tensor.insert"(%item, %current, %index) : (i64, tensor<?xi64>, index) -> tensor<?xi64>
          %inc = "arith.addi"(%index, %step) <{{overflowFlags = #arith.overflow<none>}}> : (index, index) -> index
          "scf.yield"(%inserted, %inc) : (tensor<?xi64>, index) -> ()
        }}, {{
          "scf.yield"(%current, %index) : (tensor<?xi64>, index) -> ()
        }}) : (i1) -> (tensor<?xi64>, index)
        "scf.yield"(%new, %new_index) : (tensor<?xi64>, index) -> ()
    }}) : (index, index, index, tensor<?xi64>, index) -> (tensor<?xi64>, index)
    %dim = "tensor.dim"(%result, %lo) : (tensor<?xi64>, index) -> index
    %nonnegative = "arith.cmpi"(%dim, %lo) <{{predicate = 5 : i64}}> : (index, index) -> i1
    "cf.assert"(%nonnegative) <{{msg = "count is nonnegative"}}> : (i1) -> ()
    %within = "arith.cmpi"(%dim, %hi) <{{predicate = 3 : i64}}> : (index, index) -> i1
    "cf.assert"(%within) <{{msg = "count is bounded"}}> : (i1) -> ()
    "func.return"(%result) : (tensor<?xi64>) -> ()
  }}) : () -> ()
}}) : () -> ()
"""


def _source_scatter(extent: int = 4) -> str:
    source = _source(extent).replace(
        f"function_type = (tensor<{extent}xi1>, tensor<{extent}xi64>) -> tensor<?xi64>",
        f"function_type = (tensor<{extent}xi1>, tensor<{extent}xi64>) -> tensor<{extent}xi64>",
    )

    return source.replace(
        '    "func.return"(%result) : (tensor<?xi64>) -> ()',
        f"""    %scatter_init = "tensor.empty"() : () -> tensor<{extent}xi64>
    %scatter, %scatter_count = "scf.for"(%lo, %hi, %step, %scatter_init, %lo) ({{
      ^bb4(%j: index, %dest: tensor<{extent}xi64>, %cursor: index):
        %selected = "tensor.extract"(%mask, %j) : (tensor<{extent}xi1>, index) -> i1
        %next_dest, %next_cursor = "scf.if"(%selected) ({{
          %read = "tensor.extract"(%result, %cursor) : (tensor<?xi64>, index) -> i64
          %written = "tensor.insert"(%read, %dest, %j) : (i64, tensor<{extent}xi64>, index) -> tensor<{extent}xi64>
          %incremented = "arith.addi"(%cursor, %step) <{{overflowFlags = #arith.overflow<none>}}>
            : (index, index) -> index
          "scf.yield"(%written, %incremented) : (tensor<{extent}xi64>, index) -> ()
        }}, {{
          "scf.yield"(%dest, %cursor) : (tensor<{extent}xi64>, index) -> ()
        }}) : (i1) -> (tensor<{extent}xi64>, index)
        "scf.yield"(%next_dest, %next_cursor) : (tensor<{extent}xi64>, index) -> ()
    }}) : (index, index, index, tensor<{extent}xi64>, index) -> (tensor<{extent}xi64>, index)
    "func.return"(%scatter) : (tensor<{extent}xi64>) -> ()""",
    )


def _source_internal_cast(extent: int = 4) -> str:
    """A dynamic Boolean compaction consumed by a shape-preserving cast."""
    source = _source(extent)
    source = source.replace(
        f"function_type = (tensor<{extent}xi1>, tensor<{extent}xi64>) -> tensor<?xi64>",
        f"function_type = (tensor<{extent}xi1>, tensor<{extent}xi1>) -> tensor<?xi64>",
    ).replace(f"%data: tensor<{extent}xi64>", f"%data: tensor<{extent}xi1>")
    source = source.replace(
        f'%item = "tensor.extract"(%data, %i) : (tensor<{extent}xi64>, index) -> i64',
        f'%item = "tensor.extract"(%data, %i) : (tensor<{extent}xi1>, index) -> i1',
    ).replace(
        '%inserted = "tensor.insert"(%item, %current, %index) : (i64, tensor<?xi64>, index) -> tensor<?xi64>',
        '%inserted = "tensor.insert"(%item, %current, %index) : (i1, tensor<?xi1>, index) -> tensor<?xi1>',
    )
    source = source.replace("tensor<?xi64>", "tensor<?xi1>")
    source = source.replace(
        f"function_type = (tensor<{extent}xi1>, tensor<{extent}xi1>) -> tensor<?xi1>",
        f"function_type = (tensor<{extent}xi1>, tensor<{extent}xi1>) -> tensor<?xi64>",
    )
    return source.replace(
        '    "func.return"(%result) : (tensor<?xi1>) -> ()',
        """    %cast_dim = "tensor.dim"(%result, %lo) : (tensor<?xi1>, index) -> index
    %cast_init = "tensor.empty"(%cast_dim) : (index) -> tensor<?xi64>
    %cast = "linalg.generic"(%result, %cast_init) <{
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = [#linalg.iterator_type<parallel>],
      operandSegmentSizes = array<i32: 1, 1>}> ({
      ^bb4(%bit: i1, %old_cast: i64):
        %wide_cast = "arith.extui"(%bit) : (i1) -> i64
        "linalg.yield"(%wide_cast) : (i64) -> ()
    }) : (tensor<?xi1>, tensor<?xi64>) -> tensor<?xi64>
    "func.return"(%cast) : (tensor<?xi64>) -> ()""",
    )


def _prove(source: str, *, bits: int = 32, wrong_inventory_digest: bool = False) -> dict:
    module = mq.parse(source)
    parsed = tuple(mq.walk(module))
    digest = sha256(source.encode()).hexdigest()
    inventory = {
        "capture_sha256": "0" * 64 if wrong_inventory_digest else digest,
        "capture_normalization": {"output_sha256": digest},
        "n_operations": len(parsed),
        "signatures": [
            {"mlir_operation": mq.op_name(op), "count": 1, "ordinals": [ordinal]} for ordinal, op in enumerate(parsed)
        ],
    }
    return control.prove_bounded_assertions(
        module, inventory, raw_sha256=digest, normalized_sha256=digest, index_bits=bits
    )


def test_non_operation_owner_refuses_as_unsupported_source_chain() -> None:
    from xdsl.dialects.builtin import i64
    from xdsl.ir import Block

    argument = Block(arg_types=[i64]).args[0]
    with pytest.raises(ValueError, match="source bounded-control proof: expected tensor.extract"):
        control._named(argument, "tensor.extract")

    module = mq.parse(_source())
    extract = next(op for op in mq.walk(module) if mq.op_name(op) == "tensor.extract")
    assert control._named(extract.results[0], "tensor.extract") is extract


@pytest.mark.parametrize("extent", [4, 7])
def test_exact_unsigned_mask_count_guards_are_source_tautologies(extent: int) -> None:
    result = _prove(_source(extent))
    assert result["status"] == control.PENDING
    assert result["count"] == 2
    assert [guard["count_interval"] for guard in result["guards"]] == [[0, extent], [0, extent]]
    assert result["scope"] == control.SCOPE
    assert result["index_width_status"] == "caller_premise_unverified_by_source_proof"


@pytest.mark.parametrize("extent", [4, 7])
def test_complete_mask_compaction_and_scatter_proves_only_exact_index_ordinals(extent: int) -> None:
    proof = _prove(_source_scatter(extent))
    assert len(proof["index_data_support"]) == 1
    assert proof["index_data_refusals"] == []
    chain = proof["index_data_support"][0]
    assert chain["extent"] == extent
    for operation, ordinals in (
        ("arith.index_cast", [chain["cast_ordinal"]]),
        ("arith.addi", chain["add_ordinals"]),
        ("arith.cmpi", [guard["compare_ordinal"] for guard in proof["guards"]]),
    ):
        assert control.assertion_row_proven(
            {"mlir_operation": operation, "count": len(ordinals), "ordinals": ordinals}, proof
        )
        assert not control.assertion_row_proven(
            {"mlir_operation": operation, "count": len(ordinals) + 1, "ordinals": [*ordinals, 999]}, proof
        )


@pytest.mark.parametrize("extent", [4, 7])
def test_internal_boolean_compaction_cast_proves_exact_cursor_and_allocation(extent: int) -> None:
    proof = _prove(_source_internal_cast(extent))
    assert proof["count"] == 2
    assert len(proof["internal_compaction_support"]) == 1
    chain = proof["internal_compaction_support"][0]
    assert chain["extent"] == extent
    for operation, ordinals in (
        ("arith.index_cast", [chain["cast_ordinal"]]),
        ("arith.addi", [chain["add_ordinal"]]),
    ):
        assert control.assertion_row_proven(
            {"mlir_operation": operation, "count": len(ordinals), "ordinals": ordinals}, proof
        )
    assert all(guard["count_interval"] == [0, extent] for guard in proof["guards"])


@pytest.mark.parametrize(
    "changed",
    [
        lambda s: s.replace('"tensor.extract"(%mask, %i)', '"tensor.extract"(%data, %i)'),
        lambda s: s.replace("value = 1 : index", "value = 2 : index"),
        lambda s: s.replace('"tensor.extract"(%data, %i)', '"tensor.extract"(%data, %index)'),
        lambda s: s.replace('"tensor.empty"(%cast_dim)', '"tensor.empty"(%dim)'),
        lambda s: s.replace(
            '"arith.extui"(%bit) : (i1) -> i64\n        "linalg.yield"(%wide_cast)',
            '"arith.extsi"(%bit) : (i1) -> i64\n        "linalg.yield"(%wide_cast)',
        ),
    ],
)
def test_internal_compaction_near_misses_never_discharge_cursor(changed) -> None:
    try:
        proof = _prove(changed(_source_internal_cast()))
    except ValueError:
        return
    assert proof["internal_compaction_support"] == []
    # A cast that was tried and refused is recorded beside the admitted chains, never absent.
    assert proof["internal_compaction_refusals"] and all(
        type(row["cast_ordinal"]) is int and row["reason"] for row in proof["internal_compaction_refusals"]
    )


def test_internal_compaction_requires_selected_byte_span_and_linked_identity() -> None:
    from copy import deepcopy

    narrow = _prove(_source_internal_cast(), bits=5)
    assert narrow["internal_compaction_support"] == [] and narrow["internal_compaction_refusals"]
    proof = _prove(_source_internal_cast(), bits=64)
    selected = _selected_index()
    proof["selected_index_observation"] = deepcopy(selected)
    proof["index_bits"] = selected["index_bits"]
    source = {
        "source_sha256": proof["raw_source_sha256"],
        "normalized_source_sha256": proof["normalized_source_sha256"],
        "selected_index_observation": deepcopy(selected),
    }
    control.attach_source_record(source, proof)
    linked = {"capture_tree_sha256": "c" * 64, "elf_sha256": "e" * 64, "candidate_tree_sha256": "f" * 64}
    actual = {**selected, "effective_pipeline": _effective_pipeline()}
    entry = {**linked, "index_lowering": actual}
    control.link_selected_build(source, {"output": {"index_lowering": actual}}, linked)
    assert control.linked_selected_build_complete(source, entry, "f" * 64)
    assert not control.linked_selected_build_complete(source, {**entry, "elf_sha256": "0" * 64}, "f" * 64)
    assert not control.linked_selected_build_complete(source, entry, "0" * 64)
    for mutation in (
        lambda item: item.pop("internal_compaction_support"),
        lambda item: item["internal_compaction_support"][0].update(add_ordinal=True),
        lambda item: item["internal_compaction_support"][0].update(extent=4.0),
        lambda item: item["internal_compaction_support"][0].update(
            cast_dim_ordinal=item["internal_compaction_support"][0]["loop_ordinal"]
        ),
    ):
        changed = deepcopy(source)
        mutation(changed["bounded_control_support"])
        assert not control.linked_selected_build_complete(changed, entry, "f" * 64)


def test_reviewed_dynamic_cast_body_requires_the_exact_linked_compaction_bound() -> None:
    from copy import deepcopy

    from merlin_experiments.phase1.feedback import private_linalg_support as linalg

    from merlin.targetgen.host_capabilities import admit_host_operation

    text = _source_internal_cast(7)
    module = mq.parse(text)
    parsed = tuple(mq.walk(module))
    digest = sha256(text.encode()).hexdigest()
    selected = _selected_index()
    proof = _prove(text, bits=64)
    proof["selected_index_observation"] = deepcopy(selected)
    ordinal = proof["internal_compaction_support"][0]["cast_linalg_ordinal"]
    row = {"mlir_operation": "linalg.generic", "count": 1, "ordinals": [ordinal]}
    policy = {
        "host": {
            "package_sha256": "a" * 64,
            "capability_spec_sha256": "b" * 64,
            "dtype_strategy": "int8_w8a8",
            "capability_spec": {
                "schema": "merlin.host_capabilities.v1",
                "status": "reviewed",
                "compiler": {"package_sha256": "a" * 64, "dtype_strategy": "int8_w8a8"},
                "operations": [
                    {
                        "id": "neutral_dynamic_extension",
                        "ops": ["linalg.generic"],
                        "placement": "host",
                        "signature": {
                            "ordered_operand_dtypes": ["i1", "i64"],
                            "ordered_result_dtypes": ["i64"],
                            "ranks": [1],
                        },
                        "source_body": {
                            "schema": "merlin.dynamic_boolean_cast_body.v1",
                            "operation": "i1_to_i64_extui",
                        },
                    }
                ],
                "evidence": {"scope": "synthetic placement only; no numerical review"},
            },
        }
    }
    observed = {
        "family": "cast",
        "ordered_operand_dtypes": ["i1", "i64"],
        "ordered_result_dtypes": ["i64"],
        "rank": 1,
    }
    assert admit_host_operation(policy, row, observed)["status"] == "unknown"
    admission = admit_host_operation(policy, row, observed, source_operations=(parsed[ordinal],))
    assert admission["status"] == "admitted"
    witness = linalg.begin(digest, digest, len(parsed), selected)
    with pytest.raises(ValueError, match="internal compaction"):
        linalg.record(witness, row, admission, parsed, {ordinal: row})
    linalg.record(witness, row, admission, parsed, {ordinal: row}, control_proof=proof)
    source = {
        "source_sha256": digest,
        "normalized_source_sha256": digest,
        "n_source_operations": len(parsed),
        "selected_index_observation": deepcopy(selected),
        "linalg_host_support": witness,
    }
    control.attach_source_record(source, proof)
    linked = {"capture_tree_sha256": "c" * 64, "elf_sha256": "e" * 64, "candidate_tree_sha256": "f" * 64}
    actual = {**selected, "effective_pipeline": _effective_pipeline()}
    entry = {"source_sha256": digest, **linked, "index_lowering": actual}
    with pytest.raises(ValueError, match="identical linked control witness"):
        linalg.link(source, actual, linked)
    control.link_selected_build(source, {"output": {"index_lowering": actual}}, linked)
    linalg.link(source, actual, linked)
    assert linalg.linked_complete(source, entry, "f" * 64)
    for mutation in (
        lambda value: value.pop("bounded_control_support"),
        lambda value: value["linalg_host_support"]["occurrences"][0]["dynamic_bound"].update(extent=8),
        lambda value: value["linalg_host_support"]["occurrences"][0].update(shape_source="ambient"),
        lambda value: value["bounded_control_support"]["internal_compaction_support"][0].update(extent=8),
        lambda value: value["bounded_control_support"]["selected_index_observation"].update(index_bits=32),
    ):
        changed = deepcopy(source)
        mutation(changed)
        assert not linalg.linked_complete(changed, entry, "f" * 64)
    assert not linalg.linked_complete(source, {**entry, "elf_sha256": "0" * 64}, "f" * 64)


def _different_scatter_mask(source: str) -> str:
    source = source.replace(
        '    %scatter_init = "tensor.empty"()',
        '    %other_mask = "tensor.empty"() : () -> tensor<4xi1>\n    %scatter_init = "tensor.empty"()',
    )
    return source.replace('"tensor.extract"(%mask, %j)', '"tensor.extract"(%other_mask, %j)')


def _add_two(source: str) -> str:
    source = source.replace(
        '    %result, %taken = "scf.for"',
        '    %two = "arith.constant"() <{value = 2 : index}> : () -> index\n    %result, %taken = "scf.for"',
    )
    return source.replace('"arith.addi"(%index, %step)', '"arith.addi"(%index, %two)')


@pytest.mark.parametrize(
    "changed",
    [
        _different_scatter_mask,
        _add_two,
        lambda source: source.replace(
            '    %scatter_init = "tensor.empty"()',
            '    %escape = "arith.index_cast"(%taken) : (index) -> i64\n    %scatter_init = "tensor.empty"()',
        ),
        lambda source: source.replace(
            '    %scatter_init = "tensor.empty"()',
            '    %unguarded = "tensor.extract"(%result, %hi) : (tensor<?xi64>, index) -> i64\n'
            '    %scatter_init = "tensor.empty"()',
        ),
        lambda source: source.replace(
            '    %scatter_init = "tensor.empty"()',
            '    %unguarded = "tensor.insert"(%zero64, %result, %hi) : '
            "(i64, tensor<?xi64>, index) -> tensor<?xi64>\n"
            '    %scatter_init = "tensor.empty"()',
        ),
    ],
)
def test_index_data_support_refuses_nearby_cursor_or_mask_changes(changed) -> None:
    proof = _prove(changed(_source_scatter()))
    assert proof["count"] == 2  # The independent assertion proof remains intact.
    assert proof["index_data_support"] == []
    # A refused data chain is recorded, never indistinguishable from no chain.
    assert len(proof["index_data_refusals"]) == 1
    assert type(proof["index_data_refusals"][0]["cast_ordinal"]) is int
    assert proof["index_data_refusals"][0]["reason"].startswith("source bounded-control proof:")


def test_guard_predicate_result_must_not_feed_another_control_or_data_user() -> None:
    source = _source_scatter().replace(
        '    "cf.assert"(%nonnegative)',
        '    %other = "arith.xori"(%nonnegative, %nonnegative) : (i1, i1) -> i1\n    "cf.assert"(%nonnegative)',
    )
    with pytest.raises(ValueError, match="assertion predicate has another consumer"):
        _prove(source)


def test_compaction_ordinals_and_hashes_remain_bound_to_the_linked_build() -> None:
    from copy import deepcopy

    proof = _prove(_source_scatter(), bits=64)
    selected = _selected_index()
    proof["selected_index_observation"] = deepcopy(selected)
    proof["index_bits"] = selected["index_bits"]
    source = {
        "source_sha256": proof["raw_source_sha256"],
        "normalized_source_sha256": proof["normalized_source_sha256"],
        "selected_index_observation": deepcopy(selected),
        "n_bounded_control_assertions": proof["count"],
        "n_bounded_index_support_operations": 3,
        "n_internal_mask_compactions": 0,
        "bounded_control_support": proof,
    }
    linked = {"capture_tree_sha256": "c" * 64, "elf_sha256": "e" * 64, "candidate_tree_sha256": "f" * 64}
    actual = {**deepcopy(selected), "effective_pipeline": _effective_pipeline()}
    entry = {**linked, "index_lowering": actual}
    control.link_selected_build(source, {"output": {"index_lowering": actual}}, linked)
    assert control.linked_selected_build_complete(source, entry, "f" * 64)
    for mutation in ("raw_source_sha256", "normalized_source_sha256"):
        changed = deepcopy(source)
        changed["bounded_control_support"][mutation] = "0" * 64
        assert not control.linked_selected_build_complete(changed, entry, "f" * 64)
    changed = deepcopy(source)
    changed["bounded_control_support"]["index_data_support"][0]["add_ordinals"] = [7, 7]
    assert not control.linked_selected_build_complete(changed, entry, "f" * 64)
    assert not control.linked_selected_build_complete(source, {**entry, "elf_sha256": "0" * 64}, "f" * 64)


def _different_loop_shape(source: str) -> str:
    source = source.replace(
        '    %hi = "arith.constant"()',
        '    %other = "tensor.empty"(%step) : (index) -> tensor<?xi64>\n    %hi = "arith.constant"()',
    )
    return source.replace('"scf.yield"(%current, %index)', '"scf.yield"(%other, %index)')


@pytest.mark.parametrize(
    ("changed", "reason"),
    [
        (lambda s: s.replace("value = 4 : index", "value = 3 : index"), "not a tautology"),
        (lambda s: s.replace('"arith.cmpi"(%dim, %hi)', '"arith.cmpi"(%hi, %dim)'), "expected arith.constant"),
        (lambda s: s.replace("predicate = 3 : i64", "predicate = 4 : i64"), "unsupported assertion predicate"),
        (lambda s: s.replace("value = 0 : i64", "value = 1 : i64"), "count does not start at zero"),
        (
            lambda s: s.replace("affine_map<(d0) -> (d0)>", "affine_map<(d0) -> (0)>", 1),
            "mask conversion changes element indexing",
        ),
        (lambda s: s.replace('"arith.extui"(%bit)', '"arith.extsi"(%bit)'), "unsigned Boolean extension"),
        (lambda s: s.replace("#arith.overflow<none>}> : (i64", "#arith.overflow<nsw>}> : (i64"), "overflow semantics"),
        (lambda s: s.replace("tensor<4xi1>", "tensor<4xi8>"), "invalid typed source|expected i1 tensor"),
        (_different_loop_shape, "loop can change"),
    ],
)
def test_near_miss_refuses(changed, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        _prove(changed(_source()))


def test_selected_index_width_is_required_and_bounded() -> None:
    with pytest.raises(ValueError, match="no caller-supplied index width premise"):
        _prove(_source(), bits=0)
    with pytest.raises(ValueError, match="may not fit selected signed index width"):
        _prove(_source(7), bits=3)


def test_source_inventory_digest_must_match_parsed_source() -> None:
    with pytest.raises(ValueError, match="source inventory differs"):
        _prove(_source(), wrong_inventory_digest=True)


def _selected_index() -> dict:
    return {
        "schema": "merlin.selected-index-lowering.v1",
        "compiler_requested": "synthetic-clang",
        "compiler_resolved": "/synthetic/clang",
        "compiler_sha256": "a" * 64,
        "cross_flags": ["--target=synthetic", "-march=synthetic"],
        "data_layout": "e-p:64:64",
        "index_bits": 64,
        "scope": "synthetic test fixture; no compiler executed",
    }


def _effective_pipeline(bits: int = 64) -> str:
    return ",".join(
        f"{name}{{index-bitwidth={bits}}}"
        for name in (
            "convert-index-to-llvm",
            "convert-arith-to-llvm",
            "finalize-memref-to-llvm",
            "convert-func-to-llvm",
            "convert-cf-to-llvm",
        )
    )


def test_only_complete_proven_assertion_rows_are_dischargeable() -> None:
    proof = _prove(_source())
    ordinals = [guard["assert_ordinal"] for guard in proof["guards"]]
    row = {"mlir_operation": "cf.assert", "count": 2, "ordinals": ordinals}
    assert control.assertion_row_proven(row, proof)
    assert not control.assertion_row_proven(row, None)
    compare = [guard["compare_ordinal"] for guard in proof["guards"]]
    assert control.assertion_row_proven({**row, "mlir_operation": "arith.cmpi", "ordinals": compare}, proof)
    assert not control.assertion_row_proven({**row, "ordinals": ordinals + [ordinals[0]], "count": 3}, proof)
    assert not control.assertion_row_proven({**row, "ordinals": ordinals + [999], "count": 3}, proof)


def test_linked_control_proof_requires_exact_prepared_compiler_and_elf() -> None:
    from copy import deepcopy

    selected = _selected_index()
    source = {
        "source_sha256": "d" * 64,
        "normalized_source_sha256": "b" * 64,
        "selected_index_observation": deepcopy(selected),
        "n_bounded_control_assertions": 2,
        "n_bounded_index_support_operations": 0,
        "n_internal_mask_compactions": 0,
        "bounded_control_support": {
            "status": control.PENDING,
            "count": 2,
            "guards": [{"assert_ordinal": 4, "compare_ordinal": 3}, {"assert_ordinal": 8, "compare_ordinal": 7}],
            "index_bits": 64,
            "raw_source_sha256": "d" * 64,
            "normalized_source_sha256": "b" * 64,
            "selected_index_observation": deepcopy(selected),
            "index_data_support": [],
            "internal_compaction_support": [],
        },
    }
    linked = {"capture_tree_sha256": "c" * 64, "elf_sha256": "e" * 64, "candidate_tree_sha256": "f" * 64}
    actual = {**deepcopy(selected), "effective_pipeline": _effective_pipeline()}
    receipt = {"output": {"index_lowering": actual}}
    entry = {**linked, "index_lowering": actual}
    assert control.link_selected_build(source, receipt, linked) == actual
    assert control.linked_selected_build_complete(source, entry, "f" * 64)
    for changed in (
        {**deepcopy(actual), "cross_flags": ["--target=other"]},
        {
            **deepcopy(actual),
            "effective_pipeline": _effective_pipeline().replace("index-bitwidth=64", "index-bitwidth=32", 1),
        },
        {**deepcopy(actual), "effective_pipeline": _effective_pipeline() + ",convert-cf-to-llvm{index-bitwidth=64}"},
        {key: value for key, value in actual.items() if key != "effective_pipeline"},
    ):
        assert not control.linked_selected_build_complete(source, {**entry, "index_lowering": changed}, "f" * 64)
    assert not control.linked_selected_build_complete(source, {**entry, "elf_sha256": "0" * 64}, "f" * 64)
    assert not control.linked_selected_build_complete(source, entry, "0" * 64)
    stale = deepcopy(source)
    stale["bounded_control_support"]["status"] = control.PENDING
    assert not control.linked_selected_build_complete(stale, entry, "f" * 64)
    for malformed in (
        [],
        [{"assert_ordinal": 4}],
        [{"assert_ordinal": 4}, {"assert_ordinal": 4}],
        [{"assert_ordinal": 4}, {"assert_ordinal": True}],
    ):
        changed = deepcopy(source)
        changed["bounded_control_support"]["status"] = control.PENDING
        changed["bounded_control_support"]["guards"] = malformed
        with pytest.raises(ValueError, match="not pending a linked build"):
            control.link_selected_build(changed, receipt, linked)
        assert not control.linked_selected_build_complete(changed, entry, "f" * 64)
    for malformed in (
        {**selected, "compiler_sha256": "invalid"},
        {**selected, "data_layout": "e-p:32:32"},
        {**selected, "cross_flags": "--target=synthetic"},
    ):
        changed = deepcopy(source)
        changed["selected_index_observation"] = malformed
        changed["bounded_control_support"]["selected_index_observation"] = malformed
        assert not control.linked_selected_build_complete(
            changed, {**entry, "index_lowering": {**malformed, "effective_pipeline": _effective_pipeline()}}, "f" * 64
        )


def test_prebuild_and_postbuild_observations_must_match_all_flags() -> None:
    selected = _selected_index()
    source = {
        "selected_index_observation": selected,
        "n_bounded_control_assertions": 0,
        "n_internal_mask_compactions": 0,
    }
    receipt = {"output": {"index_lowering": {**selected, "effective_pipeline": _effective_pipeline()}}}
    assert control.link_selected_build(source, receipt, {}) == receipt["output"]["index_lowering"]
    changed = {**selected, "cross_flags": ["--target=synthetic", "-march=synthetic", "-prepared-feature"]}
    with pytest.raises(ValueError, match="post-preparation index lowering differs"):
        control.link_selected_build(
            source, {"output": {"index_lowering": {**changed, "effective_pipeline": _effective_pipeline()}}}, {}
        )
    assert not control.linked_selected_build_complete(source, {"index_lowering": {}}, "f" * 64)
    with pytest.raises(ValueError, match="assertion roster is incomplete"):
        control.link_selected_build({**source, "n_bounded_control_assertions": 1}, receipt, {})


def test_source_ledger_discharge_is_exactly_guard_ordinals_not_index_arithmetic(tmp_path, monkeypatch) -> None:
    from merlin_experiments.phase1.feedback import private_full_models as gate
    from merlin_experiments.phase1.feedback import private_pure_stage_support as pure_stage

    from merlin.frontends.capture_normalization import normalize_capture_mlir
    from merlin.targetgen import application_inventory as ai
    from merlin.targetgen import eligibility
    from merlin.targetgen import model_coverage as mc
    from merlin.xdsl_dialects.lowering import compute_groups as cg

    capture = tmp_path / "capture"
    capture.mkdir()
    source = capture / "model.mlir"
    source.write_text(_source())
    (capture / "frontend-trace.json").write_text('{"graphs": {"prepared": {"nodes": []}}}')
    monkeypatch.setattr(
        ai, "verify_capture_receipt", lambda _: {"status": "verified_materialized", "receipt_sha256": "a" * 64}
    )
    monkeypatch.setattr(gate.bucketize_support, "verify_capture_receipt", ai.verify_capture_receipt)
    monkeypatch.setattr(cg, "form_groups", lambda *_: [])
    monkeypatch.setattr(cg, "plan", lambda *_a, **_k: {})
    monkeypatch.setattr(cg, "require_explained", lambda *_: None)
    monkeypatch.setattr(mc, "regions_from_module", lambda *_: [])
    monkeypatch.setattr(mc, "region_ops", lambda *_: [])
    monkeypatch.setattr(eligibility, "capability_map_from_contract", lambda *_: {})
    monkeypatch.setattr(pure_stage, "prove_if_empty", lambda *_a, **_k: None)
    compare_required = False

    def inventory(path, *_a, **_k):
        normalized, _ = normalize_capture_mlir(path.read_text())
        parsed = tuple(mq.walk(mq.parse(normalized)))
        signatures = []
        for ordinal, op in enumerate(parsed):
            name = mq.op_name(op)
            signatures.append(
                {
                    "mlir_operation": name,
                    "disposition": (
                        "unclassified"
                        if name == "cf.assert"
                        else "host_required"
                        if compare_required and name == "arith.cmpi"
                        else "component"
                    ),
                    "count": 1,
                    "ordinals": [ordinal],
                }
            )
        return {
            "capture_sha256": sha256(path.read_bytes()).hexdigest(),
            "capture_normalization": {"output_sha256": sha256(normalized.encode()).hexdigest()},
            "n_operations": len(parsed),
            "signatures": signatures,
        }

    monkeypatch.setattr(ai, "_application_operation_inventory", inventory)
    selected = _selected_index()
    result = gate._source_obligations(capture, "fixture", {}, {}, {}, selected)
    assert result["n_bounded_control_assertions"] == 2
    assert result["bounded_control_support"]["status"] == control.PENDING
    assert result["n_support_lowering_operations"] == 0
    with pytest.raises(ValueError):
        gate._source_obligations(capture, "fixture", {}, {}, {})
    compare_required = True
    result = gate._source_obligations(capture, "fixture", {}, {}, {}, selected)
    assert result["n_bounded_control_assertions"] == 2
