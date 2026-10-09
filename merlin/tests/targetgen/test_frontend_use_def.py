"""Complete original argument replay precedes shape-free interaction selection."""

import copy

import pytest

from merlin.targetgen.frontend_trace import _digest
from merlin.targetgen.frontend_use_def import original_use_def_semantics


def trace(*, kind="input", extent=17):
    nodes, edges = [], []

    def node(identity, op, inputs=()):
        result = identity + ":value"
        references = [{"node_id": producer, "value_id": producer + ":value"} for producer in inputs]
        nodes.append(
            {
                "id": identity,
                "op": op,
                "ordinal": len(nodes),
                "target": "aten.matmul.default",
                "args": references,
                "kwargs": {},
                "results": [{"id": result, "dtype": "int8", "shape": [extent, 19], "kind": "tensor"}],
            }
        )
        for index, producer in enumerate(inputs):
            edges.append(
                {
                    "producer_node_id": producer,
                    "producer_value_id": producer + ":value",
                    "consumer_node_id": identity,
                    "argument_path": "args/" + str(index),
                    "dtype": "int8",
                    "shape": [extent, 19],
                    "value_kind": "tensor",
                }
            )

    node("input", "placeholder")
    producer = "input"
    if kind != "input":
        node("producer", "call_function", ["input"])
        producer = "producer"
    node("left", "call_function", [producer])
    node("right", "call_function", [producer])
    node("output", "output", ["left", "right"] + ([producer] if kind == "escaped" else []))
    graph = {
        "schema": "m2m.frontend_graph.v1",
        "stage": "original",
        "status": "complete",
        "call_count": sum(node["op"] == "call_function" for node in nodes),
        "nodes": nodes,
        "edges": edges,
    }
    graph["sha256"] = _digest(graph)
    return {"schema": "m2m.frontend_trace.v1", "graphs": {"original": graph}}


def resign(document):
    graph = document["graphs"]["original"]
    graph["sha256"] = _digest({key: value for key, value in graph.items() if key != "sha256"})


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("input", ["shared_input_multiple_consumers"]),
        ("producer", ["shared_producer_multiple_consumers"]),
        ("escaped", ["publication_and_further_use", "shared_producer_multiple_consumers"]),
    ],
)
def test_exact_relations_project_only_class_presence(kind, expected):
    result = original_use_def_semantics(trace(kind=kind))
    assert list(result.interaction_classes) == expected
    public = result.public_semantics()
    assert set(public) == {"graph_sha256", "operation_semantics", "interaction_classes"}
    assert public["operation_semantics"] == ["aten.matmul.default"]
    original = result.witnesses()
    original[0]["producer"] = "changed"
    assert result.witnesses()[0]["producer"] != "changed"


def test_shapes_names_frequencies_do_not_become_source_templates():
    original = trace()
    changed = trace(extent=401)
    for node in changed["graphs"]["original"]["nodes"]:
        node["id"] = "renamed_" + node["id"]
        for result in node["results"]:
            result["id"] = "renamed_" + result["id"]
        for reference in node["args"]:
            for field in ("node_id", "value_id"):
                reference[field] = "renamed_" + reference[field]
    for edge in changed["graphs"]["original"]["edges"]:
        for field in ("producer_node_id", "producer_value_id", "consumer_node_id"):
            edge[field] = "renamed_" + edge[field]
    resign(changed)
    a, b = (original_use_def_semantics(item) for item in (original, changed))
    assert a.graph_sha256 != b.graph_sha256
    assert a.operations == b.operations and a.interaction_classes == b.interaction_classes


@pytest.mark.parametrize(
    "defect,reason",
    [
        ("wrong_edge_owner", "exact producer owner"),
        ("wrong_argument_owner", "exact producer owner"),
        ("duplicate_result", "result identity is duplicated"),
        ("missing_edge", "edge roster differs"),
        ("extra_edge", "edge roster differs"),
        ("duplicate_edge", "edge argument path is duplicated"),
        ("wrong_path", "edge roster differs"),
        ("missing_args", "complete original args"),
        ("bad_order", "exact source order"),
        ("forward_argument", "does not precede"),
        ("partial_reference", "complete concrete value references"),
        ("wrong_kind", "value kind disagrees"),
    ],
)
def test_resigned_graph_cannot_hide_incomplete_or_misowned_uses(defect, reason):
    document = trace()
    graph = document["graphs"]["original"]
    if defect == "wrong_edge_owner":
        graph["edges"][0]["producer_node_id"] = "right"
    elif defect == "wrong_argument_owner":
        graph["nodes"][1]["args"][0]["node_id"] = "right"
    elif defect == "duplicate_result":
        graph["nodes"][1]["results"].append(copy.deepcopy(graph["nodes"][0]["results"][0]))
    elif defect == "missing_edge":
        graph["edges"].pop()
    elif defect == "extra_edge":
        graph["edges"].append({**graph["edges"][0], "argument_path": "args/8"})
    elif defect == "duplicate_edge":
        graph["edges"].append(copy.deepcopy(graph["edges"][0]))
    elif defect == "wrong_path":
        graph["edges"][0]["argument_path"] = "kwargs/x"
    elif defect == "missing_args":
        del graph["nodes"][0]["args"]
    elif defect == "bad_order":
        graph["nodes"][0]["ordinal"] = 7
    elif defect == "forward_argument":
        graph["nodes"][1]["args"][0] = {"node_id": "right", "value_id": "right:value"}
    elif defect == "partial_reference":
        del graph["nodes"][1]["args"][0]["value_id"]
    elif defect == "wrong_kind":
        graph["edges"][0]["value_kind"] = "scalar"
    resign(document)
    with pytest.raises(ValueError, match=reason):
        original_use_def_semantics(document)


def test_nested_kwargs_are_reconstructed_at_the_exact_argument_path():
    document = trace()
    graph = document["graphs"]["original"]
    ref = graph["nodes"][1]["args"].pop()
    graph["nodes"][1]["kwargs"] = {"nested": [{"selected": ref}]}
    graph["edges"][0]["argument_path"] = "kwargs/nested/0/selected"
    resign(document)
    assert original_use_def_semantics(document).interaction_classes == ("shared_input_multiple_consumers",)
