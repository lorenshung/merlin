"""The merged Core ATen bundle has stable positional input and output mappings."""

from __future__ import annotations

import json

import numpy as np
import pytest

from merlin.frontends.linalg_mlir import parse_mlir_file
from merlin.targetgen.core_aten_batch import build_core_aten_batch
from merlin.targetgen.core_aten_batch_grade import grade_core_aten_batch


def test_nested_nonfinite_documents_preserve_ieee_values():
    from merlin.targetgen.core_aten_batch_grade import _complex_values, _grade_output

    values = [{"kind": "float", "value": value} for value in ("nan", "inf", "-inf", "-0.0")]
    decoded = np.asarray(_complex_values([values]), np.float64)
    assert np.isnan(decoded[0, 0])
    assert np.signbit(decoded[0, 3])
    expected = {"kind": "tensor", "dtype": "float64", "shape": [1, 4], "values": [values]}
    assert (
        _grade_output(
            decoded.tobytes(),
            {"dtype": "f64", "shape": [1, 4]},
            expected,
            {"comparison_parameters": {"equal_nan": True}},
        )["status"]
        == "pass"
    )


def test_module_symbols_are_namespaced_and_dynamic_return_is_preserved():
    from merlin.targetgen.core_aten_batch import _function_parts, _rename_ssa

    source = """builtin.module attributes {prov.example = "source"} {
      ml_program.global private mutable @state(dense<0> : tensor<i64>) : tensor<i64>
      func.func @forward(%x: tensor<?xi64>) -> tensor<?xi64> {
        %s = "ml_program.global_load"() <{global = @state}> : () -> tensor<i64>
        func.return %x : tensor<?xi64>
      }
    }"""
    names, body, returns, types, declarations = _function_parts(source)
    assert (names, returns, types) == (["%x"], ["%x"], ["tensor<?xi64>"])
    assert "@left_state" in _rename_ssa(declarations, "left_", {})
    assert "@right_state" in _rename_ssa(body, "right_", {})
    assert '"@state"' == _rename_ssa('"@state"', "left_", {})


@pytest.mark.parametrize("extent", [0, 1, 5])
def test_dynamic_grading_uses_runtime_extents(extent):
    data = np.arange(extent * 2, dtype=np.int64).reshape(extent, 2)
    record = {
        "overload": "fixture",
        "status": "bundled",
        "output_indices": [0],
        "output_abi": [{"dtype": "i64", "shape": [4, 2], "declared_shape": [-1, 2]}],
        "case": {"expected": {"kind": "tensor", "dtype": "int64", "shape": [extent, 2], "values": data.tolist()}},
    }
    assert (
        grade_core_aten_batch({"cases": [record]}, [data.tobytes()], output_shapes=[[extent, 2]])["passed_count"] == 1
    )
    assert grade_core_aten_batch({"cases": [record]}, [data.tobytes()])["passed_count"] == 0


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


def test_metadata_only_empty_result_requires_descriptor_observations():
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
    verdict = grade_core_aten_batch({"cases": [record]}, [raw])
    assert verdict["status_counts"] == {"ungradable": 1}
    assert "descriptor observations" in verdict["cases"]["aten.empty.memory_format"]["results"][0]["reason"]


def test_identical_functional_results_have_unambiguous_single_user_value(tmp_path):
    captures = tmp_path / "captures"
    directory = captures / "fixture"
    directory.mkdir(parents=True)
    (directory / "capsule.linalg.mlir").write_text(
        "builtin.module { func.func @forward(%x: tensor<2xi64>) -> (tensor<2xi64>, tensor<2xi64>) {\n"
        "func.return %x, %x : tensor<2xi64>, tensor<2xi64>\n} }"
    )
    abi = dict(dtype="i64", shape=[2])
    (directory / "capture.json").write_text(
        json.dumps({"capture_meta": {"ok": True, "opaque": 0, "input_abi": [abi], "output_abi": [abi]}})
    )
    (directory / "inputs.json").write_text("[[5, 9]]")
    (directory / "golden.json").write_text("[5, 9]")
    report = build_core_aten_batch(
        {"cases": [{"overload": "fixture", "expected": {"kind": "tensor"}}]}, captures, tmp_path / "bundle"
    )
    assert report["bundled_count"] == 1
    assert report["output_count"] == 1
    assert report["cases"][0]["result_projection"]["proof"] == "all_results_identical_ssa"


def test_bundle_carries_independent_mutable_globals(tmp_path):
    from merlin.llvmlower.pipeline import lower_to_llvm_ir
    from merlin.llvmlower.toolchain import m2m_python

    if not m2m_python().is_file():
        pytest.skip("native MLIR interpreter unavailable")
    cases = [{"overload": name, "expected": {"kind": "tensor"}} for name in ("left", "right")]
    captures = tmp_path / "captures"
    for case in cases:
        directory = captures / case["overload"]
        directory.mkdir(parents=True)
        (directory / "capsule.linalg.mlir").write_text("""builtin.module {
          ml_program.global private mutable @state(dense<0> : tensor<i64>) : tensor<i64>
          func.func @forward() -> tensor<i64> {
            %state = "ml_program.global_load"() <{global = @state}> : () -> tensor<i64>
            "ml_program.global_store"(%state) <{global = @state}> : (tensor<i64>) -> ()
            func.return %state : tensor<i64>
          }
        }""")
        (directory / "capture.json").write_text(
            json.dumps(
                {
                    "capture_meta": {
                        "ok": True,
                        "opaque": 0,
                        "input_abi": [],
                        "output_abi": [{"dtype": "i64", "shape": []}],
                    }
                }
            )
        )
        (directory / "inputs.json").write_text("[]")
        (directory / "golden.json").write_text("0")
    bundle = tmp_path / "bundle"
    report = build_core_aten_batch({"cases": cases}, captures, bundle)
    assert report["bundled_count"] == 2
    llvm = lower_to_llvm_ir((bundle / "model.mlir").read_text(), workdir=tmp_path / "lower")
    assert "@c0_state" in llvm
    assert "@c1_state" in llvm


def test_generic_reprinted_bundle_still_restores_caller_layouts(tmp_path, monkeypatch):
    """A module with private declarations is re-printed generically; layouts use the custom form."""
    from merlin.llvmlower import generic_form

    def generic(path, *args, **kwargs):
        out = tmp_path / "generic.mlir"
        out.write_text(
            '"builtin.module"() ({\n  "func.func"() <{sym_name = "forward"}> ({\n  }) : () -> ()\n}) : () -> ()\n'
        )
        return out

    monkeypatch.setattr(generic_form, "to_generic_form", generic)
    directory = tmp_path / "captures" / "aten__neg__default"
    directory.mkdir(parents=True)
    (directory / "capsule.linalg.mlir").write_text(
        "builtin.module {\n"
        "  func.func private @helper(f32) -> f32\n"
        "  func.func @forward(%0: tensor<2xf32>) -> tensor<2xf32> {\n"
        "    %1 = tensor.empty() : tensor<2xf32>\n"
        "    %2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, "
        'affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} '
        "ins(%0 : tensor<2xf32>) outs(%1 : tensor<2xf32>) {\n"
        "      ^bb0(%3: f32, %4: f32):\n"
        "        %5 = arith.negf %3 : f32\n"
        "        linalg.yield %5 : f32\n"
        "    } -> tensor<2xf32>\n"
        "    func.return %2 : tensor<2xf32>\n"
        "  }\n}\n",
        encoding="utf-8",
    )
    layout = {"dtype": "float32", "shape": [2], "stride": [1], "storage_offset": 0, "requires_grad": False}
    (directory / "capture.json").write_text(
        json.dumps(
            {
                "status": "captured_exact",
                "capture_meta": {
                    "ok": True,
                    "opaque": 0,
                    "input_abi": [{"dtype": "f32", "shape": [2]}],
                    "output_abi": [{"dtype": "f32", "shape": [2]}],
                    "result_contract": {
                        "authority": "fixture",
                        "results": [{**layout, "role": "user_output"}],
                        "inputs": [layout],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (directory / "inputs.json").write_text(json.dumps([[1.0, 2.0]]), encoding="utf-8")
    (directory / "golden.json").write_text(json.dumps([-1.0, -2.0]), encoding="utf-8")
    tensor = {"kind": "tensor", **layout, "values": [1.0, 2.0]}
    case = {
        "overload": "aten.neg.default",
        "arguments": {"args": {"kind": "tuple", "items": [tensor]}, "kwargs": {}},
        "expected": {**tensor, "values": [-1.0, -2.0]},
    }
    report = build_core_aten_batch(
        {"cases": [case], "denominator_sha256": "fixture"}, tmp_path / "captures", tmp_path / "out"
    )
    assert report["bundled_count"] == 1
    assert '"func.func"()' in (tmp_path / "out" / "model.mlir").read_text()
    boundary = json.loads((tmp_path / "out" / "semantic_io.json").read_text())
    assert boundary["post_indices"] == [1]


_CUSTOM_STATEFUL = """builtin.module {
  ml_program.global private mutable @state(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%x: tensor<2xf32>, %y: tensor<2xf32>) -> (tensor<2xf32>, tensor<i64>) {
    %s = ml_program.global_load @state : tensor<i64>
    func.return %x, %s : tensor<2xf32>, tensor<i64>
  }
}"""

_GENERIC_STATEFUL = """"builtin.module"() ({
  "ml_program.global"() {sym_name = "state", type = tensor<i64>, is_mutable, value = dense<0> : tensor<i64>, \
sym_visibility = "private"} : () -> ()
  "func.func"() <{sym_name = "forward", \
function_type = (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<i64>)}> ({
  ^bb0(%x: tensor<2xf32>, %y: tensor<2xf32>):
    %s = "ml_program.global_load"() <{global = @state}> {prov.aten = "aten.rand.default"} : () -> tensor<i64>
    "func.return"(%x, %s) : (tensor<2xf32>, tensor<i64>) -> ()
  }) : () -> ()
}) : () -> ()
"""


def test_generic_form_capture_matches_custom_form_parts():
    from merlin.targetgen.core_aten_batch import _function_parts

    custom = _function_parts(_CUSTOM_STATEFUL)
    generic = _function_parts(_GENERIC_STATEFUL)
    assert (generic[0], generic[2], generic[3]) == (custom[0], custom[2], custom[3])
    assert '"ml_program.global_load"() <{global = @state}>' in generic[1]
    assert "func.return" not in generic[1] and "^bb0" not in generic[1]
    assert 'sym_name = "state"' in generic[4] and "@forward" not in generic[4] and '"forward"' not in generic[4]


def test_generic_symbol_definitions_rename_with_their_references():
    from merlin.targetgen.core_aten_batch import _function_parts, _rename_ssa

    _, body, _, _, declarations = _function_parts(_GENERIC_STATEFUL)
    assert 'sym_name = "c0_state"' in _rename_ssa(declarations, "c0_", {})
    assert "@c0_state" in _rename_ssa(body, "c0_", {})
    assert 'sym_visibility = "private"' in _rename_ssa(declarations, "c0_", {})
    assert '"aten.rand.default"' in _rename_ssa(body, "c0_", {})


@pytest.mark.parametrize(
    "source",
    [
        _GENERIC_STATEFUL.replace('sym_name = "forward"', 'sym_name = "other"'),
        _GENERIC_STATEFUL.replace('"func.return"(%x, %s) : (tensor<2xf32>, tensor<i64>) -> ()', ""),
    ],
)
def test_generic_form_capture_refuses_missing_forward_or_return(source):
    from merlin.targetgen.core_aten_batch import _function_parts

    with pytest.raises(ValueError):
        _function_parts(source)


@pytest.mark.parametrize("declared", [False, True])
def test_complex_grading_interprets_observed_component_axis(declared):
    data = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
    record = dict(
        overload="fixture",
        status="bundled",
        output_indices=[0],
        output_abi=[dict(dtype="complex<f32>", shape=[2], declared_shape=[2, 2])],
        case=dict(
            expected=dict(
                kind="tensor",
                dtype="complex64",
                shape=[2],
                values=[dict(kind="complex", real=1, imag=2), dict(kind="complex", real=3, imag=4)],
            )
        ),
    )
    if not declared:
        del record["output_abi"][0]["declared_shape"]
    assert grade_core_aten_batch({"cases": [record]}, [data.tobytes()], output_shapes=[[2, 2]])["passed_count"] == 1


def test_frontend_layout_contract_is_rebased_with_batch_arguments(tmp_path):
    from merlin.llvmlower.result_layout import module_attribute, read_layout

    captures, bundle = tmp_path / "captures", tmp_path / "bundle"
    cases = [{"overload": name, "expected": {"kind": "tensor"}} for name in ("left", "right")]
    for case in cases:
        directory = captures / case["overload"]
        directory.mkdir(parents=True)
        contract = dict(
            inputs=[dict(shape=[3], stride=[1], storage_offset=0)],
            results=[dict(shape=[3], stride=[1], storage_offset=0, alias_inputs=[0], alias_results=[])],
        )
        source = (
            "builtin.module"
            + module_attribute(contract)
            + """ {
          func.func @forward(%x: tensor<3xf32>) -> tensor<3xf32> {
            func.return %x : tensor<3xf32>
          }
        }"""
        )
        (directory / "capsule.linalg.mlir").write_text(source)
        (directory / "capture.json").write_text(
            json.dumps(
                dict(
                    capture_meta=dict(
                        ok=True,
                        opaque=0,
                        input_abi=[dict(shape=[3], dtype="f32")],
                        output_abi=[dict(shape=[3], dtype="f32")],
                    )
                )
            )
        )
        (directory / "inputs.json").write_text("[[1, 2, 3]]")
        (directory / "golden.json").write_text("[1, 2, 3]")
    report = build_core_aten_batch(dict(cases=cases), captures, bundle)
    assert report["output_count"] == 2
    layout = read_layout((bundle / "model.mlir").read_text())
    assert [r["alias_inputs"] for r in layout["results"]] == [[0], [1]]


def test_overlapping_input_observation_borrows_readonly_storage(tmp_path):
    from merlin.llvmlower.input_layout import refuse_overlapping_input_aliases
    from merlin.llvmlower.result_layout import module_attribute, read_layout

    directory = tmp_path / "captures/fixture"
    directory.mkdir(parents=True)
    input_layout = dict(shape=[2, 3], stride=[0, 1], storage_offset=0)
    output_layout = dict(shape=[2, 3], stride=[3, 1], storage_offset=0, alias_inputs=[])
    layout = dict(inputs=[input_layout], results=[output_layout])
    source = (
        "builtin.module"
        + module_attribute(layout)
        + """ {
      func.func @forward(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
        %zero = arith.constant 0.0 : f32
        %out = tensor.splat %zero : tensor<2x3xf32>
        func.return %out : tensor<2x3xf32>
      }
    }"""
    )
    abi = dict(shape=[2, 3], dtype="f32")
    (directory / "capsule.linalg.mlir").write_text(source)
    (directory / "capture.json").write_text(
        json.dumps(
            dict(
                capture_meta=dict(
                    ok=True,
                    opaque=0,
                    input_abi=[abi],
                    output_abi=[abi],
                    result_contract=dict(
                        authority="same_conversion_exported_program",
                        inputs=[dict(input_layout, dtype="float32", requires_grad=False)],
                        results=[dict(output_layout, dtype="float32", requires_grad=False, role="user_output")],
                    ),
                )
            )
        )
    )
    (directory / "inputs.json").write_text("[[[1, 2, 3], [1, 2, 3]]]")
    (directory / "golden.json").write_text("[[0, 0, 0], [0, 0, 0]]")
    bundle = tmp_path / "bundle"
    build_core_aten_batch(
        dict(
            cases=[
                dict(
                    overload="fixture",
                    expected=dict(kind="tensor"),
                    arguments=dict(args=[dict(input_layout, kind="tensor", dtype="float32", requires_grad=False)]),
                )
            ]
        ),
        directory.parent,
        bundle,
    )
    actual = read_layout((bundle / "model.mlir").read_text())
    assert actual["inputs"] == [input_layout]
    assert actual["results"][1]["stride"] == [0, 1]
    assert actual["results"][1]["alias_inputs"] == [0]
    refuse_overlapping_input_aliases(actual)
    actual["results"][0] = dict(output_layout, alias_inputs=[0])
    refuse_overlapping_input_aliases(actual)
    actual["results"][0].update(role="user_input_mutation", input_index=0)
    with pytest.raises(ValueError, match="overlapping logical elements"):
        refuse_overlapping_input_aliases(actual)
