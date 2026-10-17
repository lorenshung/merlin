"""The recipe quantizes affine maps and movements where the unit declares them, in their exact forms.

WHY THIS EXISTS. A derived recipe quantized contractions, operand sums and window means only, so every
other operation of the int8 support set stayed in floating point on the host: scalar add/sub/mul/div,
neg, standalone relu, sub.Tensor, and every movement (clone, permute, cat, pads, scatters, fills).
``affine_map`` and ``movement`` are now recipe families, licensed by the unit's declared standalone
elementwise + contraction and movement capabilities. The quantizer annotates them (an affine map gets
its own output scale; a movement shares its input's, or keeps independent ones), and after
conversion states each in its exact integer form (``recipe_numerics``).
"""

from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir, repo_root
from merlin.targetgen import compute_units as cu
from merlin.targetgen import quant_recipe as QR
from merlin.targetgen import readout_facet as RF

_SOURCE = merlin_dir() / "python" / "merlin" / "targetgen" / "_recipe_quantizer.py"


def _facet() -> RF.ReadoutFacet:
    facts = {
        "facts": {
            "datapaths": [{"name": "input", "dtype": "i8"}, {"name": "accumulator", "dtype": "i32"}],
            "memories": [{"name": "accumulator", "bytes": 4096}],
            "interfaces": [
                {
                    "name": "register_bundle_layouts",
                    "unresolved": {},
                    "bundles": {
                        "StoreConfig": {
                            "width": 64,
                            "fields": {"out_scale": {"offset": 32, "width": None, "slot_width": 32}},
                        }
                    },
                }
            ],
        }
    }
    return RF.derive(
        "t",
        facts=facts,
        unit={"name": "u", "dtypes": ["int8"]},
        scalar_abi={
            "schema": RF.SCALAR_ABI_SCHEMA,
            "accumulator_dtype": "i32",
            "output_dtype": "i8",
            "scale_dtype": "f32",
            "clamp_min": -128,
            "clamp_max": 127,
            "provenance": {"scope": "test"},
        },
        readouts=[{"selector": "i8", "applies": ["acc_scale"]}],
    )


def _contract(*rows) -> dict:
    return {
        "compute_units": [
            {
                "name": "u",
                "kind": "systolic",
                "dtypes": ["int8"],
                "ops": ["matmul"],
                "semantic_capabilities": list(rows),
            }
        ]
    }


_CONTRACTION = {"family": "contraction", "dtypes": ["int8"]}
_ELEMENTWISE = {
    "family": "elementwise_map",
    "dtypes": ["int8"],
    "standalone_evidence": [{"fact": "accumulate_on_load", "roles": ["operand_load", "config", "readout", "commit"]}],
}
_MOVEMENT = {"family": "movement", "dtypes": ["int8"], "forms": ["copy"]}


def _recipe(contract: dict):
    (candidate,) = QR.derive_candidates(contract, [_facet()])
    return candidate.recipe


def test_declared_capabilities_license_the_families():
    recipe = _recipe(_contract(_CONTRACTION, _ELEMENTWISE, _MOVEMENT))
    assert QR.AFFINE_MAP in recipe.families and QR.MOVEMENT in recipe.families
    assert recipe.why[QR.AFFINE_MAP] and recipe.why[QR.MOVEMENT]


def test_an_undeclared_capability_licenses_nothing_and_says_so():
    recipe = _recipe(_contract(_CONTRACTION))
    assert QR.AFFINE_MAP not in recipe.families and QR.MOVEMENT not in recipe.families
    assert recipe.unquantized[QR.AFFINE_MAP] and recipe.unquantized[QR.MOVEMENT]
    fused_only = {**_ELEMENTWISE, "composed_with": ["contraction"]}
    fused_only.pop("standalone_evidence")
    assert QR.AFFINE_MAP not in _recipe(_contract(_CONTRACTION, fused_only)).families


def test_a_caller_without_capabilities_keeps_the_old_families():
    assert set(QR.derive(_facet()).families) <= {QR.CONTRACTION, QR.OPERAND_SUM, QR.WINDOW_MEAN}


def _operator_table() -> dict[str, list[str]]:
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_recipe_operators")
    table = next(n.value for n in ast.walk(function) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Dict))
    return {
        key.value: [ast.unparse(element) for element in value.elts]
        for key, value in zip(table.keys, table.values, strict=True)
    }


def test_the_quantizer_names_the_support_set_operators():
    table = _operator_table()
    affine = " ".join(table["affine_map"])
    movement = " ".join(table["movement"])
    for op in ("add.Scalar", "sub.Scalar", "mul.Scalar", "div.Scalar", "neg.default", "sub.Tensor", "relu.default"):
        assert op in affine, op
    for op in ("clone", "permute", "cat", "constant_pad_nd", "slice_scatter", "select_scatter", "repeat"):
        assert op in movement, op
    for op in ("avg_pool1d", "avg_pool3d", "adaptive_avg_pool1d", "adaptive_avg_pool3d"):
        assert op in " ".join(table["window_mean"]), op


_SCRIPT = r"""
import json
import numpy as np
import torch
from merlin.targetgen import _recipe_quantizer as quant
from merlin.targetgen import recipe_numerics as RN
recipe = {
    "schema": "quant_recipe_v1", "status": "derived",
    "families": ["contraction", "window_mean", "affine_map", "movement"],
    "weight": {"dtype": "int8", "granularity": "tensor", "symmetric": True, "quant_min": -127, "quant_max": 127},
    "activation": {"dtype": "int8", "granularity": "tensor", "symmetric": True,
                   "quant_min": -128, "quant_max": 127, "mode": "static"},
}
class Model(torch.nn.Module):
    def forward(self, x, y):
        d = torch.sub(x, y)
        s = torch.add(d, 0.75)
        n = torch.neg(s)
        c = torch.cat([n, x], dim=1)
        return torch.nn.functional.pad(c, (1, 1), value=0.0)
torch.manual_seed(0)
inputs = (torch.randn(64, 32), torch.randn(64, 32) * 0.6)
model = Model().eval()
converted = quant.apply_recipe(
    model, recipe, example_inputs=inputs, source_layer_inventory=quant.layer_inventory(model)
)
stats = converted._recipe_quantization_stats
rows = {row["operator"]: row for row in stats["integer_recipes"]}
sub = rows["aten.sub.Tensor"]
(s_a, s_b), s_out = sub["input_scales"], sub["output_scale"]
form = sub["integer_recipe"]
recipe_form = RN.IntegerRecipe(weights=tuple(form["weights"]), constant=form["constant"],
                               readout_scale=form["readout_scale"], activation=form["activation"])
codes = [torch.clamp(torch.round(t / s), -128, 127).to(torch.int64).numpy() for t, s in zip(inputs, (s_a, s_b))]
reference = np.clip(np.rint((codes[0] * s_a - codes[1] * s_b) / s_out), -128, 127)
print(json.dumps({
    "maps": stats["annotated_affine_maps"], "moves": stats["annotated_movements"],
    "operators": sorted(rows), "sub_weights": form["weights"],
    "add_constant": rows["aten.add.Tensor"]["integer_recipe"]["constant"],
    "neg_scale": rows["aten.neg.default"]["integer_recipe"]["readout_scale"],
    "pad_code": rows["aten.pad.default"].get("constant_code"),
    "refused": [row["operator"] for row in rows.values() if "refused" in row],
    "sub_max_error": int(np.abs(recipe_form.evaluate(*codes) - reference).max()),
}))
"""


def test_the_capture_quantizer_states_each_map_in_its_integer_form():
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python or not Path(python).is_file():
        pytest.skip("TorchAO capture interpreter is not configured")
    root = repo_root()
    paths = [root / "src", root / "src/merlin/targetgen", *sorted((root / "packages").glob("*/src"))]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(str(path) for path in paths)}
    result = subprocess.run([python, "-c", _SCRIPT], capture_output=True, text=True, timeout=300, env=env)
    assert result.returncode == 0, result.stdout + result.stderr[-5000:]
    import json

    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["maps"] == 3 and report["moves"] == 2
    assert report["operators"] == [
        "aten.add.Tensor",
        "aten.cat.default",
        "aten.neg.default",
        "aten.pad.default",
        "aten.sub.Tensor",
    ]
    assert report["sub_weights"][0] > 0 > report["sub_weights"][1]
    assert report["neg_scale"] < 0 and report["pad_code"] == 0 and not report["refused"]
    assert report["sub_max_error"] == 0


def test_capability_rows_reach_the_recipe_from_the_contract():
    """The working-tree contract's capabilities, parsed, are what derive_candidates hands derive."""
    from merlin.common.yaml import load_yaml

    contract = load_yaml(repo_root() / "examples/gemmini/target/contracts/target_contract.yaml")
    families = {cap.family for unit in cu.compute_units(contract) for cap in unit.semantic_capabilities}
    assert {"contraction", "movement"} <= families
