"""The production builder adapter: the core build's record in the shape the service verifies, with what
each group READS derived from the driver's own extraction, never from group numbering."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from merlin.perf import whole_model_builder as BLD
from merlin.perf import whole_model_verdict as V


def test_a_group_reads_the_groups_that_produced_its_buffers():
    model = {
        "steps": [
            {"group": 1, "in": "x", "out": "a"},
            {"group": 2, "in": "a", "out": "b"},
            {"group": 10, "lhs": "a", "rhs": "b", "out": "c"},
        ]
    }
    assert BLD._reads(model) == {"1": [], "2": ["1"], "10": ["1", "2"]}


def test_operand_scales_carry_a_stated_saturation_range_or_none():
    step = {"lhs_load": 0.5, "rhs_load": 2.0, "readout": 1.0, "relu": True}
    assert BLD._operands(step) == {"lhs_load": 0.5, "rhs_load": 2.0, "readout": 1.0, "relu": True}
    assert BLD._operands({**step, "saturate": [-8, 7]})["saturate"] == [-8, 7]
    assert BLD._operands({"lhs_load": 1}) is None


@pytest.fixture
def fake_core(tmp_path, monkeypatch):
    from merlin.perf import whole_model_build as WMB
    from merlin.perf import whole_model_open as WOS
    from merlin.runtime.backends import base as backends

    oracle = tmp_path / "oracle.json"
    oracle.write_text(
        json.dumps(
            {
                "groups": {
                    "1": {"compare": "exact", "sum": 1, "fnv1a": 2},
                    "2": {"compare": "bounded", "sum": 3, "fnv1a": 4, "bound_lsb": 1},
                },
                "argmax": 7,
                "golden_argmax": 7,
            }
        )
    )
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "memory_map.json").write_text(json.dumps({"groups": [{"group": 2, "element_bytes": 1}]}))
    elf = tmp_path / "model.elf"
    elf.write_text("elf")
    record = {
        "elf": str(elf),
        "elf_sha256": "e" * 64,
        "program": {"abi_header": {"sha256": "a" * 64}},
        "oracle": {"path": str(oracle)},
        "attribution": {
            "per_group": [
                {"group": 1, "op": "conv2d", "on": "package", "shape": "native"},
                {"group": 2, "op": "add", "on": "vendor"},
            ],
            "counts": {},
        },
    }
    calls = {}
    monkeypatch.setattr(WMB, "build", lambda *a, **k: calls.setdefault("build", k) and record)
    monkeypatch.setattr(WOS, "is_open_model", lambda capsule, target: False)
    capsule = SimpleNamespace(inputs={"x": 0}, outputs={"y": 0}, interface="i", weights_manifest="m", weights="w")
    monkeypatch.setattr(WMB, "load_model_capsule", lambda path: capsule)
    steps = [
        {"group": 1, "in": "x", "out": "a"},
        {"group": 2, "lhs": "a", "rhs": "x", "out": "b", "lhs_load": 1.0, "rhs_load": 1.0, "readout": 1.0},
    ]
    program = SimpleNamespace(
        extract=lambda *a, **k: {"steps": steps},
        _call=lambda step: "vendor_add(a, b)",
        UART=dict(V.PROTOCOL_TEMPLATES),
    )
    monkeypatch.setattr(backends, "whole_model_driver", lambda target: SimpleNamespace(program=program))
    return tmp_path, calls


def test_the_core_record_becomes_the_service_record(fake_core):
    tmp_path, calls = fake_core
    record = BLD.build(
        tmp_path / "pkg",
        target="toy",
        out_dir=tmp_path / "out",
        model_capsule="m",
        machine="board",
        header="h.h",
        prohibited_roles=["loop_descriptor"],
        phase0_recipe="recipe.yaml",
    )
    assert (
        calls["build"]["prohibited_roles"] == ["loop_descriptor"] and calls["build"]["phase0_recipe"] == "recipe.yaml"
    )
    groups = record["expectations"]["groups"]
    assert groups["2"]["compare"] == "bounded_int" and groups["2"]["inputs_from"] == ["1"]
    assert groups["2"]["output_element_bytes"] == 1 and groups["2"]["operands"]["readout"] == 1.0
    assert record["parameter_header_sha256"] == "a" * 64 and record["expectations"]["argmax"] == 7
    assert [r.get("call") for r in record["groups"]] == [None, "vendor_add"]
    V.Expectations.from_record(record["expectations"])  # the verdict admits what the builder states
