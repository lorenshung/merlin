"""Model demand uses the verified operation's family, not its routing spelling."""

# ruff: noqa: E501 -- the typed MLIR fixture preserves its exact operation serialization.

from __future__ import annotations

from types import SimpleNamespace

from merlin.targetgen import eligibility
from merlin.targetgen.capsule_source import model_accelerator_demand
from merlin.targetgen.compute_units import SemanticCapability

_TYPED_CONTRACTION = """builtin.module {
  func.func @forward(%a: tensor<4x19xi8>, %b: tensor<19x8xi8>) -> tensor<4x8xi32> {
    %zero = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %init = "tensor.splat"(%zero) : (i32) -> tensor<4x8xi32>
    %result = "linalg.generic"(%a, %b, %init) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%lhs: i8, %rhs: i8, %acc: i32):
      %lhs32 = "arith.extsi"(%lhs) : (i8) -> i32
      %rhs32 = "arith.extsi"(%rhs) : (i8) -> i32
      %product = "arith.muli"(%lhs32, %rhs32) : (i32, i32) -> i32
      %sum = "arith.addi"(%acc, %product) : (i32, i32) -> i32
      "linalg.yield"(%sum) : (i32) -> ()
    }) {prov.op = "int_matmul", prov.family = "contraction"} : (tensor<4x19xi8>, tensor<19x8xi8>, tensor<4x8xi32>) -> tensor<4x8xi32>
    func.return %result : tensor<4x8xi32>
  }
}"""


def _binding() -> SimpleNamespace:
    return SimpleNamespace(
        target="neutral_test",
        operand_dtype="int8",
        cap_dtype=lambda dtype: dtype,
        classes_for=lambda *, op, output_dtype: ["LOAD", "COMPUTE", "STORE"] if op == "int_matmul" else [],
    )


def test_typed_integer_contraction_derives_demand_from_canonical_family(monkeypatch) -> None:
    monkeypatch.setattr(
        eligibility,
        "capability_map_for_target",
        lambda target: {"contraction": SemanticCapability(family="contraction", dtypes=("int8",))},
    )
    assert model_accelerator_demand(_TYPED_CONTRACTION, _binding()) == (
        "contraction",
        ["LOAD", "COMPUTE", "STORE"],
    )


def test_unknown_source_family_cannot_create_accelerator_demand(monkeypatch) -> None:
    monkeypatch.setattr(
        eligibility,
        "capability_map_for_target",
        lambda target: {"contraction": SemanticCapability(family="contraction", dtypes=("int8",))},
    )
    unknown = _TYPED_CONTRACTION.replace('prov.family = "contraction"', 'prov.family = "unregistered"')
    assert model_accelerator_demand(unknown, _binding()) == (None, [])


def test_a_contraction_label_on_noncontracting_ir_cannot_create_demand(monkeypatch) -> None:
    monkeypatch.setattr(
        eligibility,
        "capability_map_for_target",
        lambda target: {"contraction": SemanticCapability(family="contraction", dtypes=("int8",))},
    )
    noncontracting = _TYPED_CONTRACTION.replace(
        "#linalg.iterator_type<reduction>", "#linalg.iterator_type<parallel>"
    )
    assert model_accelerator_demand(noncontracting, _binding()) == (None, [])


def test_an_undeclared_dtype_cannot_create_demand(monkeypatch) -> None:
    monkeypatch.setattr(
        eligibility,
        "capability_map_for_target",
        lambda target: {"contraction": SemanticCapability(family="contraction", dtypes=("fp32",))},
    )
    assert model_accelerator_demand(_TYPED_CONTRACTION, _binding()) == (None, [])
