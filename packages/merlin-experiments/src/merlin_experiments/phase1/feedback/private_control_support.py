"""Typed source proof for bounded mask-count guards and exact cursor movement.

This proves a property of the captured MLIR only.  It neither admits a host
operation nor proves that a compiler preserved an assertion or its abort path.
``index_bits`` is a caller-supplied premise, not verified backend evidence.
The caller must independently bind the selected MLIR index width; no default
width is inferred from a host triple or ELF class.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from merlin.common import mlir_query as mq
from merlin_experiments.phase1.feedback.private_data_movement import source_inventory_by_ordinal

SCOPE = (
    "typed source interval and mask cursor/data movement under caller-supplied index-width premise only; "
    "selected width, linked effects, and numerical equivalence remain separate obligations"
)
PENDING = "source_bounded_control_pending_build"
LINKED = "source_bounded_control_linked"


def selected_build_observation(host_package: Path, catalog: Path, board: str, target: str) -> dict[str, Any]:
    """Ask the actual whole-model producer for the selected host build's width.

    This is a preflight selection, not an assertion about a later prepared
    module. The linked-build verifier requires the *entire* observation to
    match the actual post-preparation compiler receipt.
    """
    from merlin.mining.registry import load_rvv_package
    from merlin.runtime.backends.spike_model import selected_model_compiler_plan
    from merlin.runtime.boards import load_boards

    package = load_rvv_package(host_package)
    selected = load_boards(catalog).get(board)
    _need(selected is not None and selected.target == target, "selected static board does not match target")
    plan = selected_model_compiler_plan(
        backend=package.backend,
        cflags_override=list(package.cflags),
        vlen=selected.vlen if package.backend == "rvv" else None,
        features=None,
    )
    return dict(plan["observation"])


def prove_if_selected(
    module: Any,
    inventory: Mapping[str, Any],
    *,
    raw_sha256: str,
    normalized_sha256: str,
    selected_observation: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Do not infer a source index width from an absent or malformed selection."""
    if selected_observation is None:
        return None
    _need(
        selected_observation.get("schema") == "merlin.selected-index-lowering.v1"
        and type(selected_observation.get("index_bits")) is int,
        "no producer-owned selected index-width observation",
    )
    proof = prove_bounded_assertions(
        module,
        inventory,
        raw_sha256=raw_sha256,
        normalized_sha256=normalized_sha256,
        index_bits=selected_observation["index_bits"],
    )
    proof["selected_index_observation"] = dict(selected_observation)
    return proof


def assertion_row_proven(row: Mapping[str, Any], proof: Mapping[str, Any] | None) -> bool:
    """Discharge only exact guard predicates or bounded compaction instructions."""
    if proof is None or proof.get("status") != PENDING:
        return False
    operation = row.get("mlir_operation")
    ordinals = row.get("ordinals")
    guards = proof.get("guards")
    if (
        not isinstance(ordinals, list)
        or not ordinals
        or any(type(ordinal) is not int or ordinal < 0 for ordinal in ordinals)
        or len(ordinals) != len(set(ordinals))
        or row.get("count") != len(ordinals)
        or not isinstance(guards, list)
    ):
        return False
    if operation == "cf.assert":
        proven = {guard.get("assert_ordinal") for guard in guards if isinstance(guard, Mapping)}
    elif operation == "arith.cmpi":
        proven = {guard.get("compare_ordinal") for guard in guards if isinstance(guard, Mapping)}
    elif operation in {"arith.index_cast", "arith.addi"}:
        chains = proof.get("index_data_support")
        if not isinstance(chains, list):
            return False
        key = "cast_ordinal" if operation == "arith.index_cast" else "add_ordinals"
        proven = set()
        for chain in chains:
            if not isinstance(chain, Mapping):
                return False
            value = chain.get(key)
            if operation == "arith.index_cast":
                proven.add(value)
            elif isinstance(value, list):
                proven.update(value)
            else:
                return False
    else:
        return False
    return set(ordinals) <= proven


def attach_source_record(result: dict[str, Any], proof: Mapping[str, Any] | None) -> None:
    """Retain the selected premise and exact proven-assertion count in source evidence."""
    if proof is None:
        return
    result["selected_index_observation"] = dict(proof["selected_index_observation"])
    result["n_bounded_control_assertions"] = proof["count"]
    result["n_bounded_index_support_operations"] = sum(
        1 + len(chain["add_ordinals"]) for chain in proof["index_data_support"]
    )
    if proof["count"]:
        result["bounded_control_support"] = dict(proof)


def _record_matches_selected(actual: Mapping[str, Any] | None, selected: Mapping[str, Any] | None) -> bool:
    """Exact preflight/post-preparation join, including effective pass options."""
    from merlin.common.digest import is_sha256
    from merlin.llvmlower.target_data_layout import default_index_bits

    if not isinstance(actual, Mapping) or not isinstance(selected, Mapping):
        return False
    pipeline = actual.get("effective_pipeline")
    if not isinstance(pipeline, str) or not pipeline:
        return False
    if {key: value for key, value in actual.items() if key != "effective_pipeline"} != dict(selected):
        return False
    from merlin.llvmlower.pipeline import _INDEX_CONVERSION_PASSES

    bits = selected.get("index_bits")
    if (
        selected.get("schema") != "merlin.selected-index-lowering.v1"
        or not isinstance(selected.get("compiler_requested"), str)
        or not selected["compiler_requested"]
        or not isinstance(selected.get("compiler_resolved"), str)
        or not selected["compiler_resolved"]
        or not is_sha256(selected.get("compiler_sha256"))
        or not isinstance(selected.get("cross_flags"), list)
        or not selected["cross_flags"]
        or any(not isinstance(flag, str) or not flag for flag in selected["cross_flags"])
        or not isinstance(selected.get("data_layout"), str)
        or type(bits) is not int
    ):
        return False
    try:
        if default_index_bits(selected["data_layout"]) != bits:
            return False
    except ValueError:
        return False
    for name in _INDEX_CONVERSION_PASSES:
        token = f"{name}{{index-bitwidth={bits}}}"
        if pipeline.count(token) != 1 or pipeline.count(name) != 1:
            return False
    return True


def _complete_guard_roster(proof: Mapping[str, Any], source: Mapping[str, Any]) -> bool:
    from merlin.common.digest import is_sha256

    count = proof.get("count")
    guards = proof.get("guards")
    if (
        type(count) is not int
        or count < 1
        or source.get("n_bounded_control_assertions") != count
        or not isinstance(guards, list)
        or len(guards) != count
        or not is_sha256(proof.get("raw_source_sha256"))
        or proof.get("raw_source_sha256") != source.get("source_sha256")
        or not is_sha256(proof.get("normalized_source_sha256"))
        or proof.get("normalized_source_sha256") != source.get("normalized_source_sha256")
        or proof.get("selected_index_observation") != source.get("selected_index_observation")
        or type(proof.get("index_bits")) is not int
        or proof["index_bits"] < 2
        or proof.get("index_bits") != (source.get("selected_index_observation") or {}).get("index_bits")
    ):
        return False
    ordinals = [guard.get("assert_ordinal") for guard in guards if isinstance(guard, Mapping)]
    compares = [guard.get("compare_ordinal") for guard in guards if isinstance(guard, Mapping)]
    chains = proof.get("index_data_support", [])
    if not isinstance(chains, list):
        return False
    data_ordinals = []
    for chain in chains:
        if (
            not isinstance(chain, Mapping)
            or set(chain) != {"cast_ordinal", "add_ordinals", "extent"}
            or type(chain.get("cast_ordinal")) is not int
            or type(chain.get("extent")) is not int
            or not 0 < chain["extent"] < (1 << (proof["index_bits"] - 1))
        ):
            return False
        adds = chain.get("add_ordinals")
        if not isinstance(adds, list) or len(adds) != 2:
            return False
        data_ordinals.extend([chain["cast_ordinal"], *adds])
    if source.get("n_bounded_index_support_operations") != len(data_ordinals):
        return False
    return (
        len(ordinals) == count
        and all(type(ordinal) is int and ordinal >= 0 for ordinal in ordinals)
        and len(set(ordinals)) == count
        and len(compares) == count
        and all(type(ordinal) is int and ordinal >= 0 for ordinal in compares)
        and len(set(compares)) == count
        and all(type(ordinal) is int and ordinal >= 0 for ordinal in data_ordinals)
        and len(set(data_ordinals)) == len(data_ordinals)
    )


def link_selected_build(source: dict[str, Any], receipt: Mapping[str, Any], linked_build: Mapping[str, Any]) -> dict:
    """Bind source guard proof to the exact selected compiled program, never a preflight guess."""
    proof = source.get("bounded_control_support")
    if proof is None:
        if source.get("selected_index_observation") is None:
            return {}
        if source.get("n_bounded_control_assertions") != 0:
            raise ValueError("source bounded-control assertion roster is incomplete")
    elif not isinstance(proof, dict) or proof.get("status") != PENDING or not _complete_guard_roster(proof, source):
        raise ValueError("source bounded-control proof was not pending a linked build")
    selected = source.get("selected_index_observation")
    actual = (receipt.get("output") or {}).get("index_lowering")
    if not _record_matches_selected(actual, selected):
        raise ValueError("post-preparation index lowering differs from selected source premise")
    if proof is not None:
        _need(proof.get("selected_index_observation") == selected, "source guard width selection changed")
        _need(proof.get("raw_source_sha256") == source.get("source_sha256"), "source guard identity changed")
        proof["status"] = LINKED
        proof["linked_build"] = dict(linked_build)
    return dict(actual)


def linked_selected_build_complete(source: Mapping[str, Any], entry: Mapping[str, Any], candidate_sha256: str) -> bool:
    """A formal source record must retain its exact selected compiler and linked ELF."""
    selected = source.get("selected_index_observation")
    if not _record_matches_selected(entry.get("index_lowering"), selected):
        return False
    proof = source.get("bounded_control_support")
    if proof is None:
        return source.get("n_bounded_control_assertions") == 0
    if not isinstance(proof, Mapping) or proof.get("status") != LINKED or not _complete_guard_roster(proof, source):
        return False
    linked = proof.get("linked_build")
    return (
        proof.get("selected_index_observation") == selected
        and proof.get("index_bits") == selected.get("index_bits")
        and proof.get("raw_source_sha256") == source.get("source_sha256")
        and source.get("n_bounded_control_assertions") == proof.get("count")
        and isinstance(linked, Mapping)
        and linked.get("capture_tree_sha256") == entry.get("capture_tree_sha256")
        and linked.get("elf_sha256") == entry.get("elf_sha256")
        and linked.get("candidate_tree_sha256") == candidate_sha256
    )


def _need(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(f"source bounded-control proof: {reason}")


def _named(value: Any, name: str) -> Any:
    owner = value.owner
    _need(owner is not None and mq.op_name(owner) == name and value in owner.results, f"expected {name}")
    return owner


def _integer(value: Any, typ: str) -> int:
    op = _named(value, "arith.constant")
    _only_provenance(op, {"value"})
    _need(len(op.results) == 1 and str(value.type) == typ, f"expected {typ} constant")
    attribute = op.properties.get("value")
    number = getattr(getattr(attribute, "value", None), "data", None)
    _need(type(number) is int, "constant has no exact integer value")
    return number


def _shape(value: Any, typ: str) -> tuple[int, ...]:
    from xdsl.dialects.builtin import TensorType

    tensor = value.type
    _need(isinstance(tensor, TensorType) and tensor.has_static_shape(), "non-static tensor")
    _need(str(tensor.element_type) == typ, f"expected {typ} tensor")
    result = tuple(tensor.get_shape())
    _need(all(type(extent) is int and extent >= 0 for extent in result), "invalid tensor extent")
    return result


def _only_provenance(op: Any, properties: set[str]) -> None:
    _need(
        set(op.properties) == properties and all(key.startswith("prov.") for key in op.attributes),
        f"{mq.op_name(op)} carries an unrecognized semantic attribute",
    )


def _mask_count(value: Any) -> tuple[int, Any]:
    """Prove a scalar i64 sum of zero-extended i1 cells lies in [0, N]."""
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, DenseArrayBase
    from xdsl.dialects.linalg.attrs import IteratorType, IteratorTypeAttr
    from xdsl.ir.affine import AffineDimExpr

    extract = _named(value, "tensor.extract")
    _only_provenance(extract, set())
    _need(str(value.type) == "i64" and len(extract.operands) == 1, "count is not a scalar i64 extraction")
    reduce = _named(extract.operands[0], "linalg.reduce")
    _only_provenance(reduce, {"dimensions"})
    dims = reduce.properties["dimensions"]
    _need(isinstance(dims, DenseArrayBase) and tuple(dims.iter_values()) == (0,), "not a full rank-one sum")
    _need(
        len(reduce.operands) == 2 and len(reduce.results) == 1 and _shape(reduce.results[0], "i64") == (),
        "invalid count reduction",
    )
    initializer = _named(reduce.operands[1], "tensor.splat")
    _only_provenance(initializer, set())
    _need(
        _shape(reduce.operands[1], "i64") == ()
        and len(initializer.operands) == 1
        and _integer(initializer.operands[0], "i64") == 0,
        "count does not start at zero",
    )
    _need(len(reduce.regions) == 1 and len(reduce.regions[0].blocks) == 1, "non-single count reduction")
    body = reduce.regions[0].blocks[0]
    ops = list(body.ops)
    _need(
        len(body.args) == 2
        and all(str(arg.type) == "i64" for arg in body.args)
        and [mq.op_name(op) for op in ops] == ["arith.addi", "linalg.yield"],
        "count reducer is not addition",
    )
    add, yielded = ops
    _only_provenance(add, {"overflowFlags"})
    _only_provenance(yielded, set())
    _need(
        tuple(add.operands) == tuple(body.args)
        and tuple(yielded.operands) == tuple(add.results)
        and set(add.properties) == {"overflowFlags"}
        and str(add.properties["overflowFlags"]) == "#arith.overflow<none>",
        "count reducer changes addition or overflow semantics",
    )

    generic = _named(reduce.operands[0], "linalg.generic")
    _only_provenance(generic, {"indexing_maps", "iterator_types", "operandSegmentSizes"})
    _need(len(generic.operands) == 2 and len(generic.results) == 1, "not a unary mask conversion")
    source, init = generic.operands
    (extent,) = _shape(source, "i1")
    _need(
        extent > 0 and _shape(init, "i64") == (extent,) and _shape(generic.results[0], "i64") == (extent,),
        "mask count shape or dtype changed",
    )
    segments = generic.properties["operandSegmentSizes"]
    maps = generic.properties["indexing_maps"]
    iterator = generic.properties["iterator_types"]
    _need(
        isinstance(segments, DenseArrayBase)
        and tuple(segments.iter_values()) == (1, 1)
        and isinstance(maps, ArrayAttr)
        and len(maps.data) == 2
        and isinstance(iterator, ArrayAttr)
        and len(iterator.data) == 1
        and isinstance(iterator.data[0], IteratorTypeAttr)
        and iterator.data[0].data == IteratorType.PARALLEL,
        "mask conversion is not one parallel elementwise map",
    )
    for item in maps.data:
        _need(
            isinstance(item, AffineMapAttr)
            and item.data.num_dims == 1
            and item.data.num_symbols == 0
            and len(item.data.results) == 1
            and item.data.results[0] == AffineDimExpr(0),
            "mask conversion changes element indexing",
        )
    _need(len(generic.regions) == 1 and len(generic.regions[0].blocks) == 1, "non-single mask body")
    block = generic.regions[0].blocks[0]
    ops = list(block.ops)
    _need(
        len(block.args) == 2
        and str(block.args[0].type) == "i1"
        and str(block.args[1].type) == "i64"
        and [mq.op_name(op) for op in ops] == ["arith.extui", "linalg.yield"],
        "mask conversion is not unsigned Boolean extension",
    )
    extension, output = ops
    _only_provenance(extension, set())
    _only_provenance(output, set())
    _need(
        tuple(extension.operands) == (block.args[0],)
        and str(extension.results[0].type) == "i64"
        and tuple(output.operands) == tuple(extension.results),
        "mask conversion has a different value path",
    )
    _need(extent < (1 << 63), "i64 count could overflow")
    return extent, source


def _same_loop_shape(value: Any, source: Any) -> bool:
    """Prove the loop-carried tensor is returned, with only shape-preserving inserts."""
    if value is source:
        return True
    owner = value.owner
    if owner is None:
        return False
    if mq.op_name(owner) == "tensor.insert":
        _only_provenance(owner, set())
        return len(owner.operands) >= 2 and owner.results[0] is value and _same_loop_shape(owner.operands[1], source)
    if mq.op_name(owner) != "scf.if" or value not in owner.results or len(owner.regions) != 2:
        return False
    position = list(owner.results).index(value)
    _only_provenance(owner, set())
    for region in owner.regions:
        if len(region.blocks) != 1:
            return False
        body = list(region.blocks[0].ops)
        if not body or mq.op_name(body[-1]) != "scf.yield" or len(body[-1].operands) <= position:
            return False
        if not _same_loop_shape(body[-1].operands[position], source):
            return False
    return True


def _empty_shape(value: Any) -> Any:
    loop = _named(value, "scf.for")
    _only_provenance(loop, set())
    _need(len(loop.regions) == 1 and len(loop.regions[0].blocks) == 1, "non-single shape loop")
    position = list(loop.results).index(value)
    block = loop.regions[0].blocks[0]
    body = list(block.ops)
    _need(
        len(loop.operands) >= 4 + position
        and len(block.args) >= 2 + position
        and body
        and mq.op_name(body[-1]) == "scf.yield"
        and len(body[-1].operands) > position
        and _same_loop_shape(body[-1].operands[position], block.args[1 + position]),
        "loop can change the dynamic tensor shape",
    )
    empty = _named(loop.operands[3 + position], "tensor.empty")
    _only_provenance(empty, set())
    _need(
        len(empty.operands) == 1
        and str(empty.operands[0].type) == "index"
        and str(value.type) == str(empty.results[0].type)
        and str(block.args[1 + position].type) == str(value.type),
        "loop does not carry one dynamic tensor.empty shape",
    )
    return empty.operands[0]


def _uses_are(value: Any, *expected: tuple[Any, int]) -> bool:
    """Compare every SSA use, including repeated uses by one operation."""
    actual = [(id(use.operation), use.index) for use in value.uses]
    wanted = [(id(operation), index) for operation, index in expected]
    return len(actual) == len(wanted) and sorted(actual) == sorted(wanted)


def _same_flat_mask(left: Any, right: Any, extent: int) -> bool:
    """Only verified reshape chains of one static i1 tensor preserve flat order."""
    from math import prod

    def root(value: Any) -> Any:
        _need(_shape(value, "i1") and prod(_shape(value, "i1")) == extent, "mask extent changed")
        owner = value.owner
        if (
            owner is None
            or not hasattr(owner, "name")
            or mq.op_name(owner) not in {"tensor.collapse_shape", "tensor.expand_shape"}
        ):
            return value
        expected = {"reassociation"}
        if mq.op_name(owner) == "tensor.expand_shape":
            expected.add("static_output_shape")
        _only_provenance(owner, expected)
        _need(len(owner.operands) == 1 and len(owner.results) == 1, "mask reshape has another value")
        return root(owner.operands[0])

    return root(left) is root(right)


def _compaction_loop(loop: Any, mask: Any, extent: int, *, compact: Any | None = None) -> Any:
    """Prove the one-bit-at-a-time cursor and conditional tensor data movement."""
    _only_provenance(loop, set())
    _need(
        len(loop.operands) == 5
        and len(loop.results) == 2
        and len(loop.regions) == 1
        and len(loop.regions[0].blocks) == 1,
        "mask loop has another control or result path",
    )
    lower, upper, step, initial, cursor_seed = loop.operands
    _need(
        _integer(lower, "index") == 0
        and _integer(upper, "index") == extent
        and _integer(step, "index") == 1
        and _integer(cursor_seed, "index") == 0,
        "mask loop is not a unit scan over the complete extent",
    )
    _need(not list(loop.results[1].uses), "mask cursor escapes the loop")
    block = loop.regions[0].blocks[0]
    _need(len(block.args) == 3, "mask loop carries another value")
    induction, current, cursor = block.args
    body = list(block.ops)
    _need([mq.op_name(op) for op in body] == ["tensor.extract", "scf.if", "scf.yield"], "mask loop body changed")
    condition, branch, yielded = body
    for op in body:
        _only_provenance(op, set())
    _need(
        tuple(condition.operands) == (mask, induction)
        and str(condition.results[0].type) == "i1"
        and tuple(branch.operands) == tuple(condition.results)
        and _uses_are(condition.results[0], (branch, 0))
        and len(branch.results) == 2
        and tuple(yielded.operands) == tuple(branch.results)
        and len(branch.regions) == 2
        and all(len(region.blocks) == 1 for region in branch.regions),
        "mask branch does not follow the scanned bit",
    )
    taken = list(branch.regions[0].blocks[0].ops)
    missed = list(branch.regions[1].blocks[0].ops)
    _need(
        [mq.op_name(op) for op in taken] == ["tensor.extract", "tensor.insert", "arith.addi", "scf.yield"]
        and [mq.op_name(op) for op in missed] == ["scf.yield"],
        "mask cursor has another taken or missed path",
    )
    read, write, increment, then_yield = taken
    (else_yield,) = missed
    for op in taken + missed:
        _only_provenance(op, {"overflowFlags"} if op is increment else set())
    _need(
        str(increment.results[0].type) == "index"
        and tuple(increment.operands) == (cursor, step)
        and str(increment.properties["overflowFlags"]) == "#arith.overflow<none>"
        and tuple(then_yield.operands) == (write.results[0], increment.results[0])
        and tuple(else_yield.operands) == (current, cursor)
        and _uses_are(increment.results[0], (then_yield, 1))
        and _uses_are(
            cursor,
            (read if compact is not None else write, 1 if compact is not None else 2),
            (increment, 0),
            (else_yield, 1),
        ),
        "mask cursor is not conditionally incremented exactly once",
    )
    if compact is None:
        _need(
            len(read.operands) == 2
            and read.operands[1] is induction
            and _shape(read.operands[0], "i64") == (extent,)
            and tuple(write.operands) == (read.results[0], current, cursor)
            and _uses_are(induction, (condition, 1), (read, 1))
            and _uses_are(current, (write, 1), (else_yield, 0)),
            "mask compaction does not write selected data at its cursor",
        )
    else:
        _need(
            tuple(read.operands) == (compact, cursor)
            and tuple(write.operands) == (read.results[0], current, induction)
            and _uses_are(induction, (condition, 1), (write, 2))
            and _uses_are(current, (write, 1), (else_yield, 0)),
            "mask scatter does not read compacted data conditionally",
        )
    _need(
        _uses_are(read.results[0], (write, 0))
        and _uses_are(write.results[0], (then_yield, 0))
        and _uses_are(branch.results[0], (yielded, 0))
        and _uses_are(branch.results[1], (yielded, 1)),
        "mask data path has an unexpected consumer",
    )
    return increment


def _index_data_chain(dim: Any, cast: Any, count_mask: Any, extent: int, ordinals: Mapping[int, int]) -> dict[str, Any]:
    """Prove one exact bounded mask-compaction/scatter chain, not general index math."""
    count = cast.operands[0]
    _need(_uses_are(count, (cast, 0)), "mask count has another consumer")
    first = _named(dim.operands[0], "scf.for")
    empty = _named(first.operands[3], "tensor.empty")
    first_body = list(first.regions[0].blocks[0].ops)
    _need(first_body and mq.op_name(first_body[0]) == "tensor.extract", "missing compaction mask read")
    first_mask = first_body[0].operands[0]
    _need(
        tuple(empty.operands) == tuple(cast.results)
        and _uses_are(cast.results[0], (empty, 0))
        and _same_flat_mask(count_mask, first_mask, extent),
        "mask count and compaction do not share one Boolean input",
    )
    first_add = _compaction_loop(first, first_mask, extent)
    uses = list(first.results[0].uses)
    _need(
        len(uses) == 2 and any(use.operation is dim and use.index == 0 for use in uses),
        "compacted tensor has another consumer",
    )
    extracts = [use.operation for use in uses if use.operation is not dim]
    _need(len(extracts) == 1 and mq.op_name(extracts[0]) == "tensor.extract", "compacted tensor is not scattered")
    read = extracts[0]
    second = read.parent_op()
    while second is not None and mq.op_name(second) != "scf.for":
        second = second.parent_op()
    _need(second is not None and second is not first, "compacted read is not in a second scan")
    _need(
        len(second.regions) == 1 and len(second.regions[0].blocks) == 1 and len(second.operands) == 5,
        "scatter loop has another control path",
    )
    second_body = list(second.regions[0].blocks[0].ops)
    _need(len(second_body) == 3 and mq.op_name(second_body[0]) == "tensor.extract", "missing scatter mask read")
    second_mask = second_body[0].operands[0]
    _need(_same_flat_mask(count_mask, second_mask, extent), "scatter mask differs from count mask")
    _need(_shape(second.operands[3], "i64") == (extent,), "scatter destination extent changed")
    second_add = _compaction_loop(second, second_mask, extent, compact=first.results[0])
    branch = second_body[1]
    _need(
        mq.op_name(branch) == "scf.if" and read is list(branch.regions[0].blocks[0].ops)[0],
        "compacted read is not the selected scatter read",
    )
    _need(_uses_are(first.results[0], (dim, 0), (read, 0)), "compacted tensor escapes proof")
    return {
        "cast_ordinal": ordinals[id(cast)],
        "add_ordinals": [ordinals[id(first_add)], ordinals[id(second_add)]],
        "extent": extent,
    }


def prove_bounded_assertions(
    module: Any, inventory: Mapping[str, Any], *, raw_sha256: str, normalized_sha256: str, index_bits: int
) -> dict[str, Any]:
    """Prove exact source guard predicates; leave their compiled preservation pending."""
    try:
        module.verify()
    except Exception as exc:
        raise ValueError("source bounded-control proof: invalid typed source") from exc
    parsed = tuple(mq.walk(module))
    joined = source_inventory_by_ordinal(parsed, inventory, raw_sha256, normalized_sha256)
    _need(type(index_bits) is int and index_bits >= 2, "no caller-supplied index width premise")
    ordinals = {id(op): position for position, op in enumerate(parsed)}
    guards = []
    index_data_support = []
    seen_casts: set[int] = set()
    for ordinal, op in enumerate(parsed):
        if mq.op_name(op) != "cf.assert":
            continue
        _need(
            joined[ordinal].get("mlir_operation") == "cf.assert"
            and len(op.operands) == 1
            and not op.results
            and str(op.operands[0].type) == "i1",
            "malformed source assertion",
        )
        _only_provenance(op, {"msg"})
        compare = _named(op.operands[0], "arith.cmpi")
        _only_provenance(compare, {"predicate"})
        _need(
            len(compare.operands) == 2 and all(str(v.type) == "index" for v in compare.operands),
            "assertion condition is not an index comparison",
        )
        predicate_attr = compare.properties.get("predicate")
        predicate = getattr(getattr(predicate_attr, "value", None), "data", None)
        _need(type(predicate) is int and predicate in (3, 5), "unsupported assertion predicate")
        bound = _integer(compare.operands[1], "index")
        _need(
            -(1 << (index_bits - 1)) <= bound <= (1 << (index_bits - 1)) - 1,
            "assertion bound does not fit selected signed index width",
        )
        dim = _named(compare.operands[0], "tensor.dim")
        _only_provenance(dim, set())
        _need(
            len(dim.operands) == 2 and _integer(dim.operands[1], "index") == 0,
            "assertion does not check the first tensor dimension",
        )
        cast = _named(_empty_shape(dim.operands[0]), "arith.index_cast")
        _only_provenance(cast, set())
        _need(
            len(cast.operands) == 1 and str(cast.operands[0].type) == "i64" and str(cast.results[0].type) == "index",
            "count changes through an unsupported cast",
        )
        extent, count_mask = _mask_count(cast.operands[0])
        _need(extent <= (1 << (index_bits - 1)) - 1, "count may not fit selected signed index width")
        _need(
            (predicate == 5 and bound <= 0) or (predicate == 3 and bound >= extent),
            "assertion is not a tautology over the proven count interval",
        )
        _need(_uses_are(compare.results[0], (op, 0)), "assertion predicate has another consumer")
        guards.append(
            {
                "assert_ordinal": ordinal,
                "compare_ordinal": ordinals[id(compare)],
                "dim_ordinal": ordinals[id(dim)],
                "count_interval": [0, extent],
                "predicate": predicate,
                "bound": bound,
            }
        )
        if id(cast) not in seen_casts:
            seen_casts.add(id(cast))
            try:
                index_data_support.append(_index_data_chain(dim, cast, count_mask, extent, ordinals))
            except ValueError:
                # Assertion tautology does not imply a complete cursor/data proof.
                # Exact unsupported index operations remain in the source ledger.
                pass
    return {
        "status": PENDING,
        "scope": SCOPE,
        "raw_source_sha256": raw_sha256,
        "normalized_source_sha256": normalized_sha256,
        "index_bits": index_bits,
        "index_width_status": "caller_premise_unverified_by_source_proof",
        "count": len(guards),
        "guards": guards,
        "index_data_support": index_data_support,
    }
