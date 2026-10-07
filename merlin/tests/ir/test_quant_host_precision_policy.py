"""W8A8 contraction selection must not silently approximate float host math."""

from __future__ import annotations

import hashlib
import json

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower import quant_passes
from merlin.llvmlower.compilation_recipe import CompilationRecipe
from merlin.runtime.backends.zephyr_model import _prepare_model_mlir
from merlin.runtime.dispatch_runtime import (
    _normalize_model_module,
    qualify_model_transform_audit,
    record_model_transform_audit,
)
from merlin.xdsl_dialects._common import text as mlir_text
from merlin.xdsl_dialects.lowering.outline import outline_dispatches

_FLOAT_ERF = """builtin.module {
  func.func @forward(%x: tensor<2x8xf32>) -> tensor<2x8xf32> {
    %e = tensor.empty() : tensor<2x8xf32>
    %y = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%x : tensor<2x8xf32>) outs(%e : tensor<2x8xf32>) {
    ^bb(%a: f32, %o: f32):
      %r = math.erf %a : f32
      linalg.yield %r : f32
    } -> tensor<2x8xf32>
    func.return %y : tensor<2x8xf32>
  }
}"""


def _rewrite_markers(module) -> tuple[int, int]:
    rendered = mlir_text(module)
    return rendered.count("math.erf"), rendered.count('prov.rewrite = "gelu_int"')


def test_default_int8_normalizer_preserves_float_host_erf():
    module = parse_mlir_text(_FLOAT_ERF)
    _normalize_model_module(module, int8_compute=True, quant_passes=None, prequant_gather=False)
    module.verify()
    assert _rewrite_markers(module) == (1, 0)


def test_default_compiled_preparation_preserves_float_host_erf(tmp_path):
    source = tmp_path / "model.mlir"
    source.write_text(_FLOAT_ERF, encoding="utf-8")
    prepared = _prepare_model_mlir(source, tmp_path, int8_compute=True)
    rendered = prepared.read_text(encoding="utf-8")
    assert rendered.count("math.erf") == 1
    assert 'prov.rewrite = "gelu_int"' not in rendered
    policy = json.loads((tmp_path / "quantization-policy.json").read_text())
    assert policy == {
        "schema": "merlin.quantization_policy.v1",
        "passes": ["contraction_int8", "conv_int8"],
        "selection": "default",
        "prepared_mlir_sha256": hashlib.sha256(prepared.read_bytes()).hexdigest(),
    }
    _prepare_model_mlir(source, tmp_path, int8_compute=False)
    assert not (tmp_path / "quantization-policy.json").exists()


def test_deliberate_all_passes_still_rewrite_erf_at_both_callers(tmp_path):
    selected = list(quant_passes.known())
    module = parse_mlir_text(_FLOAT_ERF)
    _normalize_model_module(module, int8_compute=True, quant_passes=selected, prequant_gather=False)
    module.verify()
    assert _rewrite_markers(module) == (0, 1)

    source = tmp_path / "model.mlir"
    source.write_text(_FLOAT_ERF, encoding="utf-8")
    prepared = _prepare_model_mlir(source, tmp_path, int8_compute=True, quant_passes=selected)
    rendered = prepared.read_text(encoding="utf-8")
    assert "math.erf" not in rendered
    assert 'prov.rewrite = "gelu_int"' in rendered
    assert json.loads((tmp_path / "quantization-policy.json").read_text())["passes"] == selected


def test_historical_implicit_all_six_audit_replays_without_reinterpretation(tmp_path):
    source = tmp_path / "model.mlir"
    source.write_text(_FLOAT_ERF)
    normalized = parse_mlir_text(_FLOAT_ERF)
    _normalize_model_module(
        normalized, int8_compute=True, quant_passes=list(quant_passes.known()), prequant_gather=False
    )
    receipt = record_model_transform_audit(
        source,
        tmp_path,
        normalized,
        outline_dispatches(normalized),
        enabled=True,
        normalization_recipe={
            "int8_compute": True,
            "quant_passes": None,
            "prequant_gather": False,
            "selection_policy": "all",
        },
    )
    assert qualify_model_transform_audit(receipt)["normalization_replay"] == "matched"


def test_new_explicit_safe_audit_replays_selected_passes(tmp_path):
    source = tmp_path / "model.mlir"
    source.write_text(_FLOAT_ERF)
    normalized = parse_mlir_text(_FLOAT_ERF)
    selected = list(quant_passes.compute_passes())
    _normalize_model_module(normalized, int8_compute=True, quant_passes=selected, prequant_gather=False)
    receipt = record_model_transform_audit(
        source,
        tmp_path,
        normalized,
        outline_dispatches(normalized),
        enabled=True,
        normalization_recipe={
            "int8_compute": True,
            "quant_passes": selected,
            "prequant_gather": False,
            "selection_policy": "all",
        },
    )
    assert qualify_model_transform_audit(receipt)["normalization_replay"] == "matched"


def test_explicit_policy_is_bound_in_compilation_receipt_and_refuses_stale_input(tmp_path):
    source = tmp_path / "model.mlir"
    source.write_text(_FLOAT_ERF)
    _prepare_model_mlir(source, tmp_path, int8_compute=True)
    recipe = CompilationRecipe(tmp_path, producer=source)
    policy = tmp_path / "quantization-policy.json"
    recipe.bind_preparation("quantization_policy", policy)
    bound = recipe.record["preparation"]["quantization_policy"]
    assert bound["sha256"] == hashlib.sha256(policy.read_bytes()).hexdigest()
    assert json.loads(recipe.path.read_text())["preparation"]["quantization_policy"] == bound
    with pytest.raises(ValueError, match="absent"):
        recipe.bind_preparation("missing", tmp_path / "missing.json")
    recipe.record["commands"].append({"status": "returned"})
    policy.write_text(policy.read_text() + "\n")
    with pytest.raises(ValueError, match="preparation changed"):
        recipe.completed(source)


def test_equal_prepared_ir_still_distinguishes_selected_numerical_policy(tmp_path):
    source = tmp_path / "model.mlir"
    source.write_text(_FLOAT_ERF.replace("math.erf %a", "arith.addf %a, %a"))
    default, explicit = tmp_path / "default", tmp_path / "explicit"
    default.mkdir()
    explicit.mkdir()
    left = _prepare_model_mlir(source, default, int8_compute=True)
    right = _prepare_model_mlir(source, explicit, int8_compute=True, quant_passes=list(quant_passes.known()))
    assert left.read_bytes() == right.read_bytes()
    assert (default / "quantization-policy.json").read_bytes() != (explicit / "quantization-policy.json").read_bytes()


@pytest.mark.parametrize("selected", [["gelu_int", "gelu_int"], ["unknown"], "gelu_int"])
def test_malformed_explicit_quant_selection_refused(selected):
    with pytest.raises(ValueError, match="quantization passes"):
        quant_passes.compute_passes(selected)
