"""Pure direct-return stages must carry typed, source-joined alias evidence."""

from __future__ import annotations

import hashlib
import json

import pytest
from merlin_experiments.phase1.feedback import private_pure_stage_support as pure

from merlin.common import mlir_query as mq


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _graph(label: str, *, swapped: bool = True) -> dict:
    first, second, alias, output = [f"g:{label}:root:n{index}" for index in range(4)]

    def value(node: str, shape: list[int]) -> dict:
        return {"id": f"{node}:v0", "kind": "tensor", "dtype": "float32", "shape": shape}

    nodes = [
        {
            "id": first,
            "op": "placeholder",
            "classification": "structural",
            "target": "a",
            "args": [],
            "kwargs": {},
            "results": [value(first, [2])],
        },
        {
            "id": second,
            "op": "placeholder",
            "classification": "structural",
            "target": "b",
            "args": [],
            "kwargs": {},
            "results": [value(second, [3])],
        },
        {
            "id": alias,
            "op": "call_function",
            "classification": "aten",
            "target": "aten.alias.default",
            "args": [{"node_id": second, "value_id": f"{second}:v0"}],
            "kwargs": {},
            "results": [value(alias, [3])],
        },
        {
            "id": output,
            "op": "output",
            "classification": "structural",
            "target": "output",
            "args": [
                [
                    {"node_id": alias if swapped else first, "value_id": f"{alias if swapped else first}:v0"},
                    {"node_id": first if swapped else alias, "value_id": f"{first if swapped else alias}:v0"},
                ]
            ],
            "kwargs": {},
            "results": [],
        },
    ]
    edges = [
        {
            "producer_node_id": second,
            "producer_value_id": f"{second}:v0",
            "consumer_node_id": alias,
            "argument_path": "args/0",
            "value_kind": "tensor",
            "dtype": "float32",
            "shape": [3],
        },
        {
            "producer_node_id": alias if swapped else first,
            "producer_value_id": f"{alias if swapped else first}:v0",
            "consumer_node_id": output,
            "argument_path": "args/0/0",
            "value_kind": "tensor",
            "dtype": "float32",
            "shape": [3] if swapped else [2],
        },
        {
            "producer_node_id": first if swapped else alias,
            "producer_value_id": f"{first if swapped else alias}:v0",
            "consumer_node_id": output,
            "argument_path": "args/0/1",
            "value_kind": "tensor",
            "dtype": "float32",
            "shape": [2] if swapped else [3],
        },
    ]
    return {
        "status": "complete",
        "call_count": 1,
        "by_target": {"aten.alias.default": 1},
        "nodes": nodes,
        "edges": edges,
    }


def _fixture(tmp_path):
    capture = tmp_path / "capture"
    capture.mkdir()
    source = (
        '"builtin.module"() ({ "func.func"() <{sym_name = "forward", '
        "function_type = (tensor<2xf32>, tensor<3xf32>) -> (tensor<3xf32>, tensor<2xf32>)}> ({ "
        "^bb0(%a: tensor<2xf32>, %b: tensor<3xf32>): "
        '"func.return"(%b, %a) : (tensor<3xf32>, tensor<2xf32>) -> () '
        "}) : () -> () }) : () -> ()\n"
    )
    (capture / "model.mlir").write_text(source)
    trace = {
        "schema": "m2m.frontend_trace.v1",
        "status": "complete",
        "blockers": [],
        "graphs": {name: _graph(name) for name in ("original", "prepared", "quantized")},
        "mlir": {
            "sha256": _sha(source.encode()),
            "operations": [
                {"ordinal": 0, "operation": "builtin.module"},
                {"ordinal": 1, "operation": "func.func"},
                {"ordinal": 2, "operation": "func.return", "operand_types": ["tensor<3xf32>", "tensor<2xf32>"]},
            ],
            "source_correspondence": [
                {
                    "node_id": "g:prepared:root:n2",
                    "status": "alias",
                    "alias_of_node_id": "g:prepared:root:n1",
                    "mlir_ordinals": [],
                    "result_index": 0,
                    "reason": "identity decomposition",
                }
            ],
        },
    }
    (capture / "frontend-trace.json").write_text(json.dumps(trace))
    meta = {
        "ok": True,
        "func_name": "forward",
        "opaque": 0,
        "input_abi": [{"shape": [2], "dtype": "f32"}, {"shape": [3], "dtype": "f32"}],
        "output_abi": [{"shape": [3], "dtype": "f32"}, {"shape": [2], "dtype": "f32"}],
        "frontend_trace": {
            "status": "complete",
            "path": "frontend-trace.json",
            "sha256": _sha((capture / "frontend-trace.json").read_bytes()),
        },
    }
    (capture / "meta.json").write_text(json.dumps(meta))
    (capture / "weights.safetensors.manifest.json").write_text(
        json.dumps({"0": {"kind": "input", "name": "a"}, "1": {"kind": "input", "name": "b"}})
    )
    (capture / "input_order.json").write_text(json.dumps({"a": 0, "b": 1}))
    (capture / "weights.safetensors").write_bytes(b"empty test bundle")
    receipt = {
        "schema": "m2m.capture-receipt.v1",
        "artifacts": {
            name: {"sha256": _sha((capture / name).read_bytes())}
            for name in (
                "model.mlir",
                "frontend-trace.json",
                "meta.json",
                "weights.safetensors.manifest.json",
                "weights.safetensors",
                "input_order.json",
            )
        },
        "lifted_constants": {},
        "materialized_abi": {"complete": True, "inputs": 2, "lifted_constants": []},
    }
    (capture / "capture_receipt.json").write_text(json.dumps(receipt))
    host = {
        "selected": {
            "status": "reviewed",
            "capability_spec": {
                "status": "reviewed",
                "operations": [{"status": "reviewed", "signature": {"dtypes": ["f32"]}}],
            },
        }
    }
    return capture, host


def _prove(capture, host):
    source = capture / "model.mlir"
    module = mq.parse(source.read_text())
    return pure.prove_direct_return_source(
        capture,
        module,
        host_capabilities=host,
        raw_sha256=_sha(source.read_bytes()),
        normalized_sha256=_sha(source.read_bytes()),
        n_source_operations=len(tuple(mq.walk(module))),
    )


def test_typed_multi_input_alias_stage_is_source_proven(tmp_path):
    capture, host = _fixture(tmp_path)
    proof = _prove(capture, host)
    assert proof["status"] == pure.PENDING
    assert proof["return_to_input"] == [1, 0]
    assert proof["source_operations"] == 3


def test_direct_return_needs_exact_linked_host_image(tmp_path):
    capture, host = _fixture(tmp_path)
    proof = _prove(capture, host)
    source = {
        "source_sha256": proof["raw_source_sha256"],
        "normalized_source_sha256": proof["normalized_source_sha256"],
        "capture_receipt_sha256": proof["capture_receipt_sha256"],
        "n_source_operations": 3,
        "n_linalg_regions": 0,
        "n_groups": 0,
        "eligible_groups": [],
        "direct_return_support": proof,
    }
    entry = {
        "source_sha256": proof["raw_source_sha256"],
        "capture_tree_sha256": "a" * 64,
        "elf_sha256": "b" * 64,
        "candidate_tree_sha256": "c" * 64,
        "linked_device_groups": 0,
        "sidecar_sha256": None,
        "static_host_compute_audit": [],
    }
    assert not pure.linked_direct_return_complete(source, entry, "c" * 64)
    pure.link_direct_return(
        source,
        {"capture_tree_sha256": "a" * 64, "elf_sha256": "b" * 64, "candidate_tree_sha256": "c" * 64},
    )
    assert pure.linked_direct_return_complete(source, entry, "c" * 64)
    proof["normalized_source_sha256"] = "f" * 64
    assert not pure.linked_direct_return_complete(source, entry, "c" * 64)
    proof["normalized_source_sha256"] = source["normalized_source_sha256"]
    for field, wrong in (
        ("candidate_tree_sha256", "d" * 64),
        ("elf_sha256", "e" * 64),
        ("capture_tree_sha256", "f" * 64),
    ):
        changed = {**entry, field: wrong}
        assert not pure.linked_direct_return_complete(source, changed, "c" * 64)
    for wrong in (False, 1):
        assert not pure.linked_direct_return_complete(source, {**entry, "linked_device_groups": wrong}, "c" * 64)
    assert not pure.linked_direct_return_complete({**source, "n_groups": None}, entry, "c" * 64)
    assert not pure.linked_direct_return_complete({**source, "n_groups": -1}, entry, "c" * 64)
    assert not pure.linked_direct_return_complete({**source, "n_groups": 1}, entry, "c" * 64)
    assert not pure.linked_direct_return_complete({**source, "n_linalg_regions": None}, entry, "c" * 64)
    assert not pure.linked_direct_return_complete({**source, "capture_receipt_sha256": "f" * 64}, entry, "c" * 64)
    assert not pure.linked_direct_return_complete({**source, "capture_receipt_sha256": True}, entry, "c" * 64)
    assert not pure.linked_direct_return_complete(source, {**entry, "elf_sha256": True}, "c" * 64)
    assert not pure.linked_direct_return_complete(source, entry, True)
    assert not pure.linked_direct_return_complete(
        {key: value for key, value in source.items() if key != "direct_return_support"}, entry, "c" * 64
    )
    with pytest.raises(ValueError):
        pure.link_direct_return(
            {**source, "n_groups": 1, "direct_return_support": dict(proof, status=pure.PENDING)}, entry
        )
    with pytest.raises(ValueError):
        pure.link_direct_return(
            {**source, "direct_return_support": dict(proof, status=pure.PENDING)},
            {"capture_tree_sha256": True, "elf_sha256": "b" * 64, "candidate_tree_sha256": "c" * 64},
        )
    movement = {
        "source_sha256": "d" * 64,
        "n_source_operations": 5,
        "n_linalg_regions": 1,
        "n_groups": 0,
        "eligible_groups": [],
        "direct_return_support": None,
    }
    assert pure.linked_direct_return_complete(movement, entry, "c" * 64)
    pure.link_direct_return(movement, {"capture_tree_sha256": "a" * 64})


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_return",
        "hidden_math",
        "constant_return",
        "custom_op",
        "bad_trace",
        "opaque",
        "trace_alias_mismatch",
        "trace_output_mismatch",
        "artifact_mismatch",
        "bf16_host",
    ],
)
def test_pure_stage_near_miss_refused(tmp_path, mutation):
    capture, host = _fixture(tmp_path)
    source = capture / "model.mlir"
    trace_path = capture / "frontend-trace.json"
    meta_path = capture / "meta.json"
    if mutation == "wrong_return":
        source.write_text(
            source.read_text()
            .replace("-> (tensor<3xf32>, tensor<2xf32>)", "-> (tensor<2xf32>, tensor<3xf32>)")
            .replace(
                '"func.return"(%b, %a) : (tensor<3xf32>, tensor<2xf32>)',
                '"func.return"(%a, %b) : (tensor<2xf32>, tensor<3xf32>)',
            )
        )
    elif mutation == "hidden_math":
        source.write_text(
            source.read_text().replace(
                '"func.return"(%b, %a)',
                '"arith.addf"(%a, %a) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32> "func.return"(%b, %a)',
            )
        )
    elif mutation == "constant_return":
        source.write_text(
            source.read_text().replace(
                '"func.return"(%b, %a)',
                '%c = "arith.constant"() <{value = dense<0.000000e+00> : tensor<2xf32>}> '
                ': () -> tensor<2xf32> "func.return"(%b, %c)',
            )
        )
    elif mutation == "custom_op":
        source.write_text(
            source.read_text().replace('"func.return"(%b, %a)', '"custom.effect"() : () -> () "func.return"(%b, %a)')
        )
    elif mutation in {"bad_trace", "trace_alias_mismatch", "trace_output_mismatch"}:
        trace = json.loads(trace_path.read_text())
        if mutation == "bad_trace":
            trace["status"] = "incomplete"
        elif mutation == "trace_alias_mismatch":
            trace["mlir"]["source_correspondence"][0]["alias_of_node_id"] = "g:prepared:root:n0"
        else:
            trace["graphs"]["prepared"]["nodes"][-1]["args"][0].reverse()
        trace_path.write_text(json.dumps(trace))
    elif mutation == "opaque":
        meta = json.loads(meta_path.read_text())
        meta["opaque"] = 1
        meta_path.write_text(json.dumps(meta))
    elif mutation == "artifact_mismatch":
        meta_path.write_text(meta_path.read_text() + " ")
    else:
        host["selected"]["capability_spec"]["operations"][0]["signature"]["dtypes"] = ["bf16"]
    if mutation != "artifact_mismatch":
        receipt_path = capture / "capture_receipt.json"
        receipt = json.loads(receipt_path.read_text())
        for name in receipt["artifacts"]:
            receipt["artifacts"][name]["sha256"] = _sha((capture / name).read_bytes())
        receipt_path.write_text(json.dumps(receipt))
        if mutation in {"bad_trace", "trace_alias_mismatch", "trace_output_mismatch"}:
            meta = json.loads(meta_path.read_text())
            meta["frontend_trace"]["sha256"] = _sha(trace_path.read_bytes())
            meta_path.write_text(json.dumps(meta))
            receipt["artifacts"]["meta.json"]["sha256"] = _sha(meta_path.read_bytes())
            receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        _prove(capture, host)
