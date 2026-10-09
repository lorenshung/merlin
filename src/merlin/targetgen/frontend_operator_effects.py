"""Conservative original argument/result effects from observed typed schemas.

Source SSA and canonical schema observations must agree. This reader produces
shape-free mandatory effect classes and private exact witnesses, not physical
aliasing, allocation ownership or a claim that schemas enumerate all effects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .frontend_use_def import original_use_def_semantics


@dataclass(frozen=True)
class OriginalOperatorEffects:
    graph_sha256: str
    effect_classes: tuple[str, ...]
    witnesses_json: str
    unknowns_json: str

    def witnesses(self):
        return json.loads(self.witnesses_json)

    def unknowns(self):
        return json.loads(self.unknowns_json)

    def public_semantics(self):
        return {"graph_sha256": self.graph_sha256, "effect_classes": list(self.effect_classes)}


def _alias(row):
    alias = row["alias"]
    if alias is None:
        return set(), False
    if (
        not isinstance(alias, dict)
        or set(alias) != {"before", "after", "write"}
        or type(alias["write"]) is not bool
        or any(
            not isinstance(alias[key], list) or any(not isinstance(item, str) or not item for item in alias[key])
            for key in ("before", "after")
        )
    ):
        raise ValueError("operator effects need actual typed alias observations")
    before, after = set(alias["before"]), set(alias["after"])
    if "*" in before | after or before != after:
        raise ValueError("wildcard or changing alias sets have no supported exact effect relation")
    return before, alias["write"]


def _bindings(node, arguments, values):
    positional = [index for index, row in enumerate(arguments) if not row["kwarg_only"]]
    if len(node["args"]) > len(positional):
        raise ValueError("original source positional arguments exceed the observed schema")
    names = {row["name"]: index for index, row in enumerate(arguments)}
    if len(names) != len(arguments) or set(node["kwargs"]) - set(names):
        raise ValueError("original source keyword arguments differ from the observed schema")
    selected = {index: value for index, value in zip(positional, node["args"])}
    paths = {index: "args/" + str(offset) for offset, index in enumerate(positional[: len(node["args"])])}
    for name, value in node["kwargs"].items():
        index = names[name]
        if index in selected:
            raise ValueError("original source argument is bound twice")
        selected[index], paths[index] = value, "kwargs/" + name
    for index, row in enumerate(arguments):
        if index not in selected and not row["has_default"]:
            raise ValueError("original source required argument has no concrete binding")
        value = selected.get(index)
        if row["type"] == "Tensor":
            if not isinstance(value, dict) or set(value) != {"node_id", "value_id"} or value["value_id"] not in values:
                raise ValueError("schema tensor argument has no exact original tensor value")
            if values[value["value_id"]]["kind"] != "tensor":
                raise ValueError("schema tensor argument disagrees with original value kind")
        elif row["alias"] is not None:
            raise ValueError("non-scalar tensor aliases require an unsupported container binding")
    return selected, paths


def original_operator_effects(trace, observation):
    """Replay all original uses before joining each exact observed schema row.

    Narrow support covers direct Tensor arguments and direct Tensor returns.
    Unknown lists, conditional alias sets, unresolved schemas and unmatched
    result rosters remain explicit. A schema with no alias annotations is not
    asserted pure: random, exception, global state and other effects are outside
    this source relation.
    """
    relation = original_use_def_semantics(trace)
    graph = trace["graphs"]["original"]
    schemas = graph.get("operator_schemas", {})
    if observation.get("schema") != "merlin.native_operator_schema_observation.v1":
        raise ValueError("original effects require the fixed native observation schema")
    rows = observation.get("rows")
    if (
        not isinstance(rows, list)
        or len({row["target"] for row in rows}) != len(rows)
        or {row["target"] for row in rows} != set(relation.operations)
    ):
        raise ValueError("schema observation must cover the complete exact original call roster")
    observed = {row["target"]: row for row in rows}
    values = {result["id"]: result for node in graph["nodes"] for result in node["results"]}
    witnesses, unknowns = [], []
    for node in graph["nodes"]:
        if node["op"] not in {"call_function", "call_method", "call_module"}:
            continue
        target, found = node["target"], observed[node["target"]]
        if found["status"] != "observed":
            unknowns.append({"target": target, "reason": found.get("reason", "schema is unobserved")})
            continue
        if node["op"] != "call_function" or schemas.get(target) != found["schema"]:
            unknowns.append({"target": target, "reason": "original captured schema or call kind differs"})
            continue
        try:
            arguments, returns = found["arguments"], found["returns"]
            bindings, paths = _bindings(node, arguments, values)
            inputs = [_alias(row) for row in arguments]
            outputs = [_alias(row) for row in returns]
            if len(node["results"]) != len(returns) or any(
                row["type"] != "Tensor" or result["kind"] != "tensor" for row, result in zip(returns, node["results"])
            ):
                raise ValueError("original result roster is not the exact supported direct Tensor returns")
            local = []
            for index, (aliases, write) in enumerate(inputs):
                if aliases and index not in bindings:
                    raise ValueError("aliased input has no explicit original value")
                if write:
                    local.append(
                        {
                            "kind": "may_write_argument",
                            "node": node["id"],
                            "target": target,
                            "argument_path": paths[index],
                            "input_value": bindings[index]["value_id"],
                        }
                    )
            for result_index, (aliases, _) in enumerate(outputs):
                if aliases and not any(aliases & input_aliases for input_aliases, _ in inputs):
                    raise ValueError("aliased return has no supported original input alias relation")
                for input_index, (input_aliases, _) in enumerate(inputs):
                    if aliases & input_aliases:
                        local.append(
                            {
                                "kind": "may_alias_result",
                                "node": node["id"],
                                "target": target,
                                "argument_path": paths[input_index],
                                "input_value": bindings[input_index]["value_id"],
                                "result_value": node["results"][result_index]["id"],
                            }
                        )
            witnesses += local
        except (KeyError, TypeError, ValueError) as exc:
            unknowns.append({"target": target, "reason": str(exc)})
    return OriginalOperatorEffects(
        relation.graph_sha256,
        tuple(sorted({row["kind"] for row in witnesses})),
        json.dumps(witnesses, sort_keys=True, separators=(",", ":")),
        json.dumps(unknowns, sort_keys=True, separators=(",", ":")),
    )
