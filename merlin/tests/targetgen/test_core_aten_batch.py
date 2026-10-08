"""The merged Core ATen bundle has stable positional input and output mappings."""

from __future__ import annotations

import json

import numpy as np

from merlin.frontends.linalg_mlir import parse_mlir_file
from merlin.targetgen.core_aten_batch import build_core_aten_batch
from merlin.targetgen.core_aten_batch_grade import grade_core_aten_batch


def test_batch_inlines_clean_cases_and_keeps_unavailable_case(tmp_path):
    captures = tmp_path / "captures"
    cases = [
        {"overload": "aten.add.Tensor", "expected": {"kind": "tensor"}},
        {"overload": "aten.sub.Tensor", "expected": {"kind": "tensor"}},
        {"overload": "aten.missing.default", "expected": {"kind": "tensor"}},
    ]
    for name, op, values in (
        ("aten__add__Tensor", "arith.addf", [[1.0, 2.0], [3.0, 4.0]]),
        ("aten__sub__Tensor", "arith.subf", [[4.0, 3.0], [2.0, 1.0]]),
    ):
        directory = captures / name
        directory.mkdir(parents=True)
        (directory / "capsule.linalg.mlir").write_text(
            "builtin.module {\n"
            "  func.func @forward(%0: tensor<2xf32>) -> tensor<2xf32> {\n"
            "    %1 = tensor.empty() : tensor<2xf32>\n"
            "    %2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, "
            'affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} '
            "ins(%0 : tensor<2xf32>) outs(%1 : tensor<2xf32>) {\n"
            "      ^bb0(%3: f32, %4: f32):\n"
            f"        %5 = {op} %3, %3 : f32\n"
            "        linalg.yield %5 : f32\n"
            "    } -> tensor<2xf32>\n"
            "    func.return %2 : tensor<2xf32>\n"
            "  }\n}\n",
            encoding="utf-8",
        )
        (directory / "capture.json").write_text(
            json.dumps(
                {
                    "status": "captured_exact",
                    "capture_meta": {
                        "ok": True,
                        "opaque": 0,
                        "input_abi": [{"dtype": "f32", "shape": [2]}],
                        "output_abi": [{"dtype": "f32", "shape": [2]}],
                    },
                }
            ),
            encoding="utf-8",
        )
        (directory / "inputs.json").write_text(json.dumps([values[0]]), encoding="utf-8")
        (directory / "golden.json").write_text(json.dumps(values[1]), encoding="utf-8")
    corpus = {"cases": cases, "denominator_sha256": "fixture"}
    first = tmp_path / "first"
    second = tmp_path / "second"
    report = build_core_aten_batch(corpus, captures, first)
    build_core_aten_batch(corpus, captures, second)
    assert (report["case_count"], report["bundled_count"], report["input_count"], report["output_count"]) == (
        3,
        2,
        2,
        2,
    )
    assert [row["output_indices"] for row in report["cases"][:2]] == [[0], [1]]
    assert report["cases"][2]["status"] == "capture_unavailable"
    assert "%c0_2" in (first / "model.mlir").read_text()
    assert "%c1_2" in (first / "model.mlir").read_text()
    parse_mlir_file(first / "model.mlir")
    with np.load(first / "inputs.npz") as data:
        assert data.files == ["in0", "in1"]
        np.testing.assert_array_equal(data["in0"], [1, 2])
        np.testing.assert_array_equal(data["in1"], [4, 3])
    for filename in (
        "model.mlir",
        "inputs.npz",
        "core_aten_batch_map.json",
        "weights.safetensors.manifest.json",
        "input_order.json",
    ):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_batch_grader_uses_float_tolerance_and_keeps_all_case_failures():
    mapping = {
        "cases": [
            {
                "overload": "aten.close.default",
                "status": "bundled",
                "output_indices": [0],
                "output_abi": [{"dtype": "f32", "shape": [2]}],
                "case": {
                    "comparison_parameters": {"rtol": 1e-5, "atol": 1e-6, "equal_nan": True},
                    "expected": {"kind": "tensor", "dtype": "float32", "shape": [2], "values": [1.0, 2.0]},
                },
            },
            {
                "overload": "aten.bad.default",
                "status": "bundled",
                "output_indices": [1],
                "output_abi": [{"dtype": "i64", "shape": [1]}],
                "case": {
                    "comparison_parameters": {},
                    "expected": {"kind": "tensor", "dtype": "int64", "shape": [1], "values": [7]},
                },
            },
            {"overload": "aten.unavailable.default", "status": "capture_unavailable", "reason": "opaque"},
        ]
    }
    outputs = [np.array([1.000001, 2.0], np.float32).tobytes(), np.array([8], np.int64).tobytes()]
    verdict = grade_core_aten_batch(mapping, outputs)
    assert verdict["status_counts"] == {"capture_unavailable": 1, "mismatch": 1, "pass": 1}
    assert verdict["cases"]["aten.bad.default"]["results"][0]["actual"] == 8
    failed = grade_core_aten_batch(mapping, execution_error="compiler crash")
    assert failed["status_counts"] == {"capture_unavailable": 1, "execution_failed": 2}


def test_metadata_only_empty_result_uses_contiguous_descriptor_contract():
    record = {
        "overload": "aten.empty.memory_format",
        "status": "bundled",
        "output_indices": [0],
        "output_abi": [{"dtype": "f32", "shape": [2, 3]}],
        "case": {
            "comparison": "metadata",
            "comparison_parameters": {},
            "expected": {
                "kind": "tensor",
                "dtype": "float32",
                "shape": [2, 3],
                "stride": [3, 1],
                "requires_grad": False,
            },
        },
    }
    raw = np.zeros((2, 3), np.float32).tobytes()
    passed = grade_core_aten_batch({"cases": [record]}, [raw])
    assert passed["passed_count"] == 1
    evidence = passed["cases"]["aten.empty.memory_format"]["results"][0]["evidence"]
    assert evidence["stride"] == [3, 1]
    assert evidence["values"] == "unspecified_by_case"
    record["case"]["expected"]["stride"] = [1, 2]
    failed = grade_core_aten_batch({"cases": [record]}, [raw])
    assert failed["status_counts"] == {"mismatch": 1}
