"""Source-bound pool proof: neutral geometry and adversarial source/trace changes."""

from __future__ import annotations

import json
from hashlib import sha256
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.feedback import private_full_models as gate
from merlin_experiments.phase1.feedback.private_pool_support import (
    LINKED,
    POOL_ATEN,
    linked_pool_complete,
    prove_pool_source,
)

from merlin.common import mlir_query as mq

SOURCE_NODE = "g:prepared:root:n1"
MODEL = """"builtin.module"() ({
  "func.func"() <{sym_name = "forward", function_type = (tensor<1x1x2x6xf32>) -> tensor<1x1x1x3xf32>}> ({
  ^bb0(%0: tensor<1x1x2x6xf32>):
    %1 = "arith.constant"() <{value = 0xff800000 : f32}> : () -> f32
    %2 = "tensor.splat"(%1) : (f32) -> tensor<1x1x1x3xf32>
    %3 = "tensor.empty"() : () -> tensor<2x2xf32>
    %4 = "linalg.generic"(%0, %3, %2) <{indexing_maps = [
      affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, ((d2 * 2) + d4), ((d3 * 2) + d5))>,
      affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>,
      affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>],
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>,
                        #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>,
                        #linalg.iterator_type<reduction>, #linalg.iterator_type<reduction>],
      operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb1(%5: f32, %6: f32, %7: f32):
      %8 = "arith.cmpf"(%5, %5) <{predicate = 14 : i64, fastmath = #arith.fastmath<none>}> : (f32, f32) -> i1
      %9 = "arith.cmpf"(%5, %7) <{predicate = 2 : i64, fastmath = #arith.fastmath<none>}> : (f32, f32) -> i1
      %10 = "arith.ori"(%8, %9) : (i1, i1) -> i1
      %11 = "arith.select"(%10, %5, %7) : (i1, f32, f32) -> f32
      "linalg.yield"(%11) : (f32) -> ()
    }) {prov.aten = "aten.max_pool2d_with_indices.default", prov.region_id = "pool_0",
        prov.source_node_ids = ["g:prepared:root:n1"]} :
      (tensor<1x1x2x6xf32>, tensor<2x2xf32>, tensor<1x1x1x3xf32>) -> tensor<1x1x1x3xf32>
    "func.return"(%4) : (tensor<1x1x1x3xf32>) -> ()
  }) : () -> ()
}) : () -> ()
"""


def _hash(data: bytes) -> str:
    return sha256(data).hexdigest()


def _capture(tmp_path, model=MODEL, *, index=0, direct_tuple=False, parameters=None):
    capture = tmp_path / "capture"
    capture.mkdir()
    model_path = capture / "model.mlir"
    model_path.write_text(model, encoding="utf-8")
    module = mq.parse(model)
    ordinal = next(i for i, op in enumerate(mq.walk(module)) if mq.op_name(op) == "linalg.generic")
    generic = tuple(mq.walk(module))[ordinal]
    input_value = generic.operands[0]
    if getattr(input_value.owner, "name", None) == "tensor.insert_slice":
        input_value = input_value.owner.operands[0]
    input_shape = list(input_value.type.get_shape())
    output_shape = list(generic.results[0].type.get_shape())
    kernel, stride, padding, dilation = parameters or ((2, 2), (2, 2), (0, 0), (1, 1))
    model_sha = _hash(model.encode())
    pool = {
        "id": SOURCE_NODE,
        "op": "call_function",
        "target": POOL_ATEN,
        "args": [
            {"node_id": "g:prepared:root:n0", "value_id": "g:prepared:root:n0:v0"},
            list(kernel),
            list(stride),
            list(padding),
            list(dilation),
            False,
        ],
        "kwargs": {},
        "results": [
            {"id": f"{SOURCE_NODE}:v0", "dtype": "float32", "shape": output_shape},
            {"id": f"{SOURCE_NODE}:v1", "dtype": "int64", "shape": output_shape},
        ],
    }
    selector = {
        "id": "g:prepared:root:n2",
        "op": "call_function",
        "target": "<built-in function getitem>",
        "args": [{"node_id": SOURCE_NODE, "value_id": f"{SOURCE_NODE}:v{index}"}, index],
        "kwargs": {},
    }
    if direct_tuple:
        selector["target"] = "output"
    trace = {
        "schema": "m2m.frontend_trace.v1",
        "status": "complete",
        "graphs": {
            "prepared": {
                "status": "complete",
                "nodes": [pool, selector],
                "edges": [
                    {
                        "producer_node_id": "g:prepared:root:n0",
                        "producer_value_id": "g:prepared:root:n0:v0",
                        "consumer_node_id": SOURCE_NODE,
                        "argument_path": "args/0",
                        "dtype": "float32",
                        "shape": input_shape,
                    },
                    {
                        "producer_node_id": SOURCE_NODE,
                        "producer_value_id": f"{SOURCE_NODE}:v{index}",
                        "consumer_node_id": selector["id"],
                        "argument_path": "args/0",
                    },
                ],
            }
        },
        "mlir": {
            "sha256": model_sha,
            "source_correspondence": [{"node_id": SOURCE_NODE, "status": "lowered", "mlir_ordinals": [ordinal]}],
        },
    }
    trace_path = capture / "frontend-trace.json"
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    receipt = {
        "artifacts": {
            "model.mlir": {"sha256": model_sha},
            "frontend-trace.json": {"sha256": _hash(trace_path.read_bytes())},
        }
    }
    (capture / "capture_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    inventory = {
        "capture_sha256": model_sha,
        "n_operations": len(tuple(mq.walk(module))),
        "capture_normalization": {"output_sha256": model_sha},
        "signatures": [
            {
                "frontend_op": POOL_ATEN,
                "mlir_operation": "linalg.generic",
                "disposition": "host_required",
                "count": 1,
                "ordinals": [ordinal],
            }
        ],
    }
    return capture, module, inventory, model_sha


def _prove(tmp_path, model=MODEL, **kwargs):
    capture, module, inventory, digest = _capture(tmp_path, model, **kwargs)
    return prove_pool_source(capture, module, inventory, raw_sha256=digest, normalized_sha256=digest)


def test_value_only_ordered_reducer_is_source_bound_and_link_requires_exact_bytes(tmp_path):
    proof = _prove(tmp_path)
    assert proof["count"] == 1
    assert proof["occurrences"][0]["geometry"] == {
        "kernel": (2, 2),
        "stride": (2, 2),
        "padding": (0, 0),
        "dilation": (1, 1),
    }
    source = {
        "source_sha256": proof["raw_source_sha256"],
        "normalized_source_sha256": proof["normalized_source_sha256"],
        "capture_receipt_sha256": proof["capture_receipt_sha256"],
        "pool_value_support": proof,
    }
    entry = {"source_sha256": proof["raw_source_sha256"], "capture_tree_sha256": "a" * 64, "elf_sha256": "b" * 64}
    assert not linked_pool_complete(source, entry, "c" * 64)
    proof["status"] = LINKED
    proof["linked_build"] = {"candidate_tree_sha256": "c" * 64, "capture_tree_sha256": "a" * 64, "elf_sha256": "b" * 64}
    assert linked_pool_complete(source, entry, "c" * 64)
    proof["normalized_source_sha256"] = "f" * 64
    assert not linked_pool_complete(source, entry, "c" * 64)
    proof["normalized_source_sha256"] = source["normalized_source_sha256"]
    assert not linked_pool_complete(source, entry, "d" * 64)
    receipt_sha = proof.pop("capture_receipt_sha256")
    source.pop("capture_receipt_sha256")
    assert not linked_pool_complete(source, entry, "c" * 64)
    proof["capture_receipt_sha256"] = receipt_sha
    source["capture_receipt_sha256"] = receipt_sha
    del proof["linked_build"]["capture_tree_sha256"]
    assert not linked_pool_complete(source, entry, "c" * 64)


def test_static_padding_dilation_and_floor_geometry(tmp_path):
    padded = (
        MODEL.replace("tensor<1x1x2x6xf32>", "tensor<1x1x4x5xf32>")
        .replace("tensor<1x1x1x3xf32>", "tensor<1x1x2x3xf32>")
        .replace(
            '    %3 = "tensor.empty"()',
            '    %20 = "tensor.splat"(%1) : (f32) -> tensor<1x1x4x7xf32>\n'
            '    %21 = "tensor.insert_slice"(%0, %20) '
            "<{static_offsets = array<i64: 0, 0, 0, 1>, "
            "static_sizes = array<i64: 1, 1, 4, 5>, "
            "static_strides = array<i64: 1, 1, 1, 1>, "
            "operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> "
            ": (tensor<1x1x4x5xf32>, tensor<1x1x4x7xf32>) -> tensor<1x1x4x7xf32>\n"
            '    %3 = "tensor.empty"()',
        )
        .replace("tensor<2x2xf32>", "tensor<2x3xf32>")
        .replace('"linalg.generic"(%0, %3, %2)', '"linalg.generic"(%21, %3, %2)')
        .replace("((d2 * 2) + d4)", "(d2 + (d4 * 2))")
        .replace("((d3 * 2) + d5)", "((d3 * 2) + d5)")
        .replace(
            "(tensor<1x1x4x5xf32>, tensor<2x3xf32>, tensor<1x1x2x3xf32>)",
            "(tensor<1x1x4x7xf32>, tensor<2x3xf32>, tensor<1x1x2x3xf32>)",
        )
    )
    proof = _prove(tmp_path, padded, parameters=((2, 3), (1, 2), (0, 1), (2, 1)))
    assert proof["occurrences"][0]["geometry"]["dilation"] == (2, 1)


def test_legacy_maximumf_body_is_not_ordered_torch_pool(tmp_path):
    old_body = MODEL[MODEL.index('      %8 = "arith.cmpf"') : MODEL.index("    }) {prov.aten")]
    stale = MODEL.replace(
        old_body, '      %8 = "arith.maximumf"(%5, %7) : (f32, f32) -> f32\n      "linalg.yield"(%8) : (f32) -> ()\n'
    )
    with pytest.raises(ValueError, match="reducer is not exact ordered select"):
        _prove(tmp_path, stale)


@pytest.mark.parametrize(
    "model",
    [
        MODEL.replace('"arith.cmpf"(%5, %5)', '"arith.cmpf"(%7, %7)'),
        MODEL.replace("predicate = 2 : i64", "predicate = 1 : i64"),
        MODEL.replace("fastmath = #arith.fastmath<none>", "fastmath = #arith.fastmath<nnan>"),
        MODEL.replace('"arith.select"(%10, %5, %7)', '"arith.select"(%10, %7, %5)'),
        MODEL.replace('"linalg.yield"(%11)', '"linalg.yield"(%5)'),
        MODEL.replace("((d2 * 2) + d4)", "((d2 * 2) + (d4 * 2))"),
        MODEL.replace("tensor<2x2xf32>", "tensor<2x3xf32>"),
        MODEL.replace("0xff800000", "0x00000000"),
    ],
)
def test_source_reducer_map_window_and_initializer_changes_fail_closed(tmp_path, model):
    with pytest.raises(ValueError, match="source max-pool proof"):
        _prove(tmp_path, model)


@pytest.mark.parametrize("kwargs", [{"index": 1}, {"direct_tuple": True}])
def test_indices_or_full_tuple_are_not_value_only(tmp_path, kwargs):
    with pytest.raises(ValueError, match="source max-pool proof"):
        _prove(tmp_path, **kwargs)


@pytest.mark.parametrize("change", [lambda node: node.pop("args"), lambda node: node.__setitem__("args", [])])
def test_pool_trace_without_an_input_argument_is_refused(tmp_path, change):
    capture, module, inventory, digest = _capture(tmp_path)
    trace_path = capture / "frontend-trace.json"
    trace = json.loads(trace_path.read_text())
    change(next(node for node in trace["graphs"]["prepared"]["nodes"] if node["id"] == SOURCE_NODE))
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    receipt = json.loads((capture / "capture_receipt.json").read_text())
    receipt["artifacts"]["frontend-trace.json"]["sha256"] = _hash(trace_path.read_bytes())
    (capture / "capture_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="pool trace has no input argument"):
        prove_pool_source(capture, module, inventory, raw_sha256=digest, normalized_sha256=digest)


def test_boolean_ordinal_is_not_an_integer_source_occurrence(tmp_path):
    capture, module, inventory, digest = _capture(tmp_path)
    inventory["signatures"][0]["ordinals"] = [True]
    with pytest.raises(ValueError, match="occurrence ordinal is not an in-range integer"):
        prove_pool_source(capture, module, inventory, raw_sha256=digest, normalized_sha256=digest)


def test_accelerator_pool_needs_no_host_trace_proof_but_all_ordinals_are_accounted(tmp_path):
    capture, module, inventory, digest = _capture(tmp_path)
    ordinal = inventory["signatures"][0]["ordinals"][0]
    inventory["signatures"][0]["disposition"] = "hardware_admitted"
    (capture / "frontend-trace.json").unlink()
    proof = prove_pool_source(capture, module, inventory, raw_sha256=digest, normalized_sha256=digest)
    assert proof["count"] == 0
    assert proof["occurrences"] == []
    assert proof["device_ordinals"] == [ordinal]
    inventory["signatures"][0]["disposition"] = "unclassified"
    with pytest.raises(ValueError, match="exact host or device occurrences"):
        prove_pool_source(capture, module, inventory, raw_sha256=digest, normalized_sha256=digest)


def test_accelerator_pool_remains_in_eligible_group_and_operation_admission(tmp_path, monkeypatch):
    from merlin.frontends.capture_normalization import normalize_capture_mlir
    from merlin.targetgen import application_inventory as ai
    from merlin.targetgen import eligibility
    from merlin.targetgen import model_coverage as mc
    from merlin.targetgen import operation_accounting as oa
    from merlin.xdsl_dialects.lowering import compute_groups as cg

    capture, _, inventory, digest = _capture(tmp_path)
    normalized, _ = normalize_capture_mlir((capture / "model.mlir").read_text(encoding="utf-8"))
    parsed = mq.parse(normalized)
    operations = tuple(mq.walk(parsed))
    inventory["n_operations"] = len(operations)
    inventory["capture_normalization"]["output_sha256"] = _hash(normalized.encode())
    pool_ordinal = next(i for i, op in enumerate(operations) if mq.op_name(op) == "linalg.generic")
    pool_row = {**inventory["signatures"][0], "ordinals": [pool_ordinal], "disposition": "hardware_admitted"}
    nested_components = {"arith.cmpf", "arith.ori", "arith.select", "linalg.yield"}
    inventory["signatures"] = [
        pool_row
        if ordinal == pool_ordinal
        else {
            "mlir_operation": mq.op_name(op),
            "disposition": "component" if mq.op_name(op) in nested_components else "structural",
            "count": 1,
            "ordinals": [ordinal],
        }
        for ordinal, op in enumerate(operations)
    ]
    monkeypatch.setattr(
        ai, "verify_capture_receipt", lambda _path: {"status": "verified_materialized", "receipt_sha256": "f" * 64}
    )
    monkeypatch.setattr(gate.bucketize_support, "verify_capture_receipt", ai.verify_capture_receipt)
    monkeypatch.setattr(ai, "_application_operation_inventory", lambda *_args, **_kwargs: inventory)
    monkeypatch.setattr(mc, "regions_from_module", lambda _module: [object()])
    monkeypatch.setattr(
        mc, "region_ops", lambda module: [op for op in mq.walk(module) if mq.op_name(op) == "linalg.generic"]
    )
    monkeypatch.setattr(
        cg,
        "form_groups",
        lambda module, _target: [
            SimpleNamespace(
                index=0,
                placement="accelerator",
                root=next(op for op in mq.walk(module) if mq.op_name(op) == "linalg.generic"),
                members=tuple(op for op in mq.walk(module) if mq.op_name(op) == "linalg.generic"),
            )
        ],
    )
    monkeypatch.setattr(cg, "plan", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(cg, "require_explained", lambda _plan: None)
    monkeypatch.setattr(eligibility, "capability_map_from_contract", lambda _contract: {})
    monkeypatch.setattr(
        eligibility, "is_eligible", lambda *_args, **_kwargs: SimpleNamespace(undetermined=False, eligible=True)
    )
    admitted = []

    def admit(row, **_kwargs):
        admitted.append(row)
        return {
            "accelerator_admission": {"status": "admitted"},
            "host_admission": {"status": "unsupported", "reviewed": False},
        }

    monkeypatch.setattr(oa, "admit_operation_row", admit)
    proof = gate._source_obligations(capture, "neutral", {}, {}, {})
    assert proof["source_sha256"] == digest
    assert proof["pool_value_support"]["count"] == 0
    assert proof["pool_value_support"]["device_ordinals"] == [pool_ordinal]
    assert proof["eligible_groups"] == [0]
    assert admitted == [pool_row]
