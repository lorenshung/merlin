"""Native invalid-result NaN bits require an explicit source policy."""

import json

import numpy as np
from mlir_fixtures import forward, generic

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen.core_aten_components import prepare_integer_components
from merlin.targetgen.native_dtype_domain import component_semantics


def division_module(literal):
    prelude = f"%zero=arith.constant {literal} : f32\n%divisor=tensor.splat %zero : tensor<1xf32>\n"
    ins = [("%x", "tensor<1xi32>"), ("%divisor", "tensor<1xf32>")]
    body = "%w=arith.sitofp %a : i32 to f32\n%v=arith.divf %w,%b : f32"
    return forward("%x: tensor<1xi32>", "tensor<1xf32>", prelude + generic(ins, "tensor<1xf32>", body))


def test_certified_integer_conversion_zero_division():
    proof = component_semantics(division_module("0.0"))
    assert proof["status"] == "proven"
    assert proof["method"] == "closed_typed_pointwise_IEEE_composition_v1"


def test_uncertified_binary64_zero_divisor_falls_back_before_provider(tmp_path):
    source = str(division_module("0.0"))
    source = source.replace("i32", "f64").replace("f32", "f64")
    source = source.replace("arith.sitofp %a : f64 to f64", "arith.negf %a : f64")
    module = parse_mlir_text(source)
    original = str(module) + "\n"
    (tmp_path / "model.mlir").write_text(original)
    np.savez(tmp_path / "inputs.npz", in0=np.array([0], np.float64))

    class Provider:
        def admit_numeric_schedule(self, *args, **kwargs):
            raise AssertionError("undefined native NaN bits cannot request a device proof")

    assert (
        prepare_integer_components(
            tmp_path,
            dict(cases=[dict(input_abi=[dict(dtype="f64")])]),
            dict(spec={}, host={}),
            Provider(),
            None,
            module,
            limb_attribute="test.limb",
        )
        is None
    )
    assert (tmp_path / "model.mlir").read_text() == original
    refusal = json.loads((tmp_path / "exact-dtype-refusal.json").read_text())
    assert refusal["code"] == "native_binary64_arithmetic_form_unproved"
