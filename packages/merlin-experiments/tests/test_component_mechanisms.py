"""Actual shared-product and convolution programs through normal generation."""

import copy
from types import SimpleNamespace

import pytest
import test_component_coverage as coverage_fixtures
import test_component_generation as generation_fixtures
import yaml
from merlin_experiments.phase0 import component_coverage, generation
from merlin_experiments.phase0.component_source_witnesses import verify

from merlin.targetgen import capsule_results, capsule_source, golden_store
from merlin.targetgen.component_sources import parameters, source_effects

independent = generation_fixtures.independent


def typed_contraction(options):
    contract = yaml.safe_load(options["capability_contract"].read_bytes())
    contract["compute_units"][0]["accumulate"] = [{"in": "int8", "weight": "int8", "acc": "i32"}]
    generation_fixtures.write(options["capability_contract"], contract)


@pytest.mark.parametrize("tile", [2, 3])
def test_multiple_products_share_operands_with_complete_publication(independent, tile):
    typed_contraction(independent)
    coverage_fixtures.update_hardware(independent, tile=tile)
    m, k, n = tile + 1, tile + 2, tile + 3
    program = {
        "inputs": [
            {"name": name, "role": role, "shape": shape, "dtype": "operand"}
            for name, role, shape in [
                ("A", "input", [m, k]),
                ("B", "input", [m, k]),
                ("W", "weight", [k, n]),
                ("V", "weight", [k, n]),
            ]
        ],
        "nodes": [
            {"name": name, "op": "matmul", "inputs": args}
            for name, args in [("P", ["A", "W"]), ("Q", ["B", "W"]), ("R", ["A", "V"])]
        ],
        "outputs": [{"name": name.lower(), "value": name} for name in ("P", "Q", "R")],
    }
    kinds = ["multiple_consumers", "immutable_reuse", "output_publication"]
    obligation = {
        "id": "shared_products",
        "mandatory": True,
        "cohort": "functional_guard",
        "operations": ["contraction"],
        "effects": kinds,
        "expectation": "admitted_program",
        "frontend": "mlir",
        "base": {"op": "component_program", "kind": "model_slice", "program": program},
        "axes": {},
        "interactions": [],
    }
    coverage_fixtures.plan_for(
        independent,
        [obligation],
        effects=[
            {"id": kind, "status": "reviewed", "kind": kind, "basis": "explicit immutable independent tensor inputs"}
            for kind in kinds
        ],
    )
    generation.generate_target("fixture", **independent)
    report = component_coverage.verify_report(independent["output_root"])
    member = report["obligations"][0]["members"][0]
    root = independent["output_root"] / member["member"]
    cap = yaml.safe_load((root / "capsule.yaml").read_bytes())
    from merlin.targetgen.capsule_inputs import materialize_capsule_leaves

    leaves = materialize_capsule_leaves(cap)
    expected = {
        name: leaves[a].matmul(leaves[w]).to_list()
        for name, a, w in [("p", "A", "W"), ("q", "B", "W"), ("r", "A", "V")]
    }
    assert golden_store.load_golden(root)["outputs"] == expected
    assert member["output_roster"] == ["p", "q", "r"]
    assert cap["component_program"]["uses"]["A"] == cap["component_program"]["uses"]["W"] == 2


@pytest.mark.parametrize("tile", [2, 3])
def test_convolution_window_tails_cross_selected_byte_frontier(independent, tile):
    typed_contraction(independent)
    contract = yaml.safe_load(independent["capability_contract"].read_bytes())
    contract["compute_units"][0]["ops"].append("conv2d")
    contract["compute_units"][0]["semantic_capabilities"][0]["ranks"] = [2, 4]
    generation_fixtures.write(independent["capability_contract"], contract)
    coverage_fixtures.update_hardware(independent, tile=tile, extra={"source_window_bytes": 256})
    software = yaml.safe_load(independent["software_spec"].read_bytes())
    software["operations"]["contraction"]["ranks"] = [2, 4]
    generation_fixtures.write(independent["software_spec"], software)
    obligation = {
        "id": "conv_frontier",
        "mandatory": True,
        "cohort": "functional_guard",
        "operations": ["contraction"],
        "effects": [],
        "expectation": "admitted_program",
        "frontend": "mlir",
        "base": {
            "op": "conv2d",
            "kind": "layer",
            "ci": 2,
            "Wimg": 9,
            "kh": 3,
            "kw": 2,
            "stride": [2, 1],
            "padding": [1, 0, 0, 1],
            "dilation": [1, 2],
        },
        "axes": {
            "N": {"kind": "extent", "values": ["tile+1"]},
            "Himg": {
                "kind": "resource",
                "declaration": {
                    "derive": "declared_resource_boundary",
                    "resource": "source_window",
                    "capacity_fact": ["source_window_bytes"],
                    "reservation_facts": [],
                    "quantum": "tile",
                    "tail_offsets": [-1, 1],
                    "allocations": [
                        {"name": "IFM", "shape": [1, "Himg", 9, 2], "dtype": "operand"},
                        {"name": "W", "shape": [12, "N"], "dtype": "operand"},
                    ],
                },
            },
        },
        "interactions": [],
    }
    coverage_fixtures.plan_for(independent, [obligation])
    generation.generate_target("fixture", **independent)
    report = component_coverage.verify_report(independent["output_root"])
    row = report["obligations"][0]
    points = row["resource_boundaries"]["Himg"]["points"]
    assert any(point["fits"] for point in points) and any(not point["fits"] for point in points)
    heights = []
    for member in row["members"]:
        root = independent["output_root"] / member["member"]
        cap = yaml.safe_load((root / "capsule.yaml").read_bytes())
        attrs = cap["operation"]["attributes"]
        assert attrs["stride"] == [2, 1] and attrs["dilation"] == [1, 2] and attrs["padding"] == [1, 0, 0, 1]
        heights.append(next(inp["shape"][1] for inp in cap["inputs"] if inp["name"] == "IFM"))
        from merlin.targetgen.capsule_inputs import materialize_capsule_leaves

        leaves = materialize_capsule_leaves(cap)
        image, weights = leaves["IFM"], leaves["W"]
        _, h, w, channels = image.shape
        out_h, out_w = (h + 1 - 3) // 2 + 1, w + 1 - 3 + 1
        expected = []
        # Independent source-window sum; no runtime im2col/golden helper.
        for oy in range(out_h):
            for ox in range(out_w):
                pixel = []
                for output in range(tile + 1):
                    total = 0
                    for ky in range(3):
                        for kx in range(2):
                            iy, ix = oy * 2 - 1 + ky, ox + 2 * kx
                            if 0 <= iy < h and 0 <= ix < w:
                                for channel in range(channels):
                                    value = image.data[(iy * w + ix) * channels + channel]
                                    weight = weights.data[((ky * 2 + kx) * channels + channel) * (tile + 1) + output]
                                    total += value * weight
                    pixel.append(total)
                expected.append(pixel)
        assert golden_store.load_golden(root)["outputs"] == {"Y0": expected}
    assert set(heights) == {point["extent"] for point in points}
    assert any(height % tile for height in heights)


def captured_results():
    arrays = [[[0.0, -0.0, 0.5, 1.5, 2.5, -1.5, 200.0, -200.0]]]
    codes = [[[0.0, 0.0, 0.0, 2.0, 2.0, -2.0, 127.0, -128.0]]]
    return SimpleNamespace(
        meta={"output_abi": [{"shape": [1, 8], "dtype": "f32"}] * 3 + [{"shape": [1, 1], "dtype": "f32"}]},
        linalg_mlir="""builtin.module {
          func.func @forward(%x: tensor<1x8xf32>, %w: tensor<8x1xf32>) ->
          (tensor<1x8xf32>, tensor<1x8xf32>, tensor<1x8xf32>, tensor<1x1xf32>) {
             %out = tensor.empty() : tensor<1x1xf32>
             func.return %x, %x, %x, %out : tensor<1x8xf32>, tensor<1x8xf32>, tensor<1x8xf32>, tensor<1x1xf32>
          }
        }""",
        golden=[arrays[0], codes[0], codes[0], [[1.0]]],
    )


@pytest.mark.parametrize("change", ["abi", "golden", "shape", "names", "dtype"])
def test_captured_complete_publication_refuses_incomplete_or_mismatched_results(change):
    art, entry = captured_results(), {}
    assert list(capsule_results.publication(art, entry)) == ["Y0", "Y1", "Y2", "Y3"]
    if change == "abi":
        art.meta["output_abi"] = []
    elif change == "golden":
        art.golden.pop()
    elif change == "shape":
        art.golden[3] = [[1.0, 2.0]]
    elif change == "names":
        entry["outs"] = ["same"] * 4
    else:
        art.meta["output_abi"][0] = {"shape": [1, 8], "dtype": "i8"}
    with pytest.raises(ValueError):
        capsule_results.publication(art, entry)


def test_independent_quantizer_source_witness_checks_ties_clamps_and_producer_zero():
    art = captured_results()
    outputs = capsule_results.publication(art, {})
    attrs = {"outs": list(outputs), **parameters({"dtype": "f32"})}
    cap = {"operation": {"op": "producer_quantizer_observer", "attributes": attrs}}
    golden = {
        "outputs": outputs,
        "golden_source": "host_torch_eager",
        "oracle_provenance": {"inputs": {"X": {"shape": [1, 8], "decoded": art.golden[0][0]}}},
    }
    witness = verify(cap, golden)
    assert witness["ties"] == 4 and witness["clamps"] == 2 and witness["negative_zero_producer_elements"] == 1
    assert set(witness["effects"]) == {"quantizer_tie_input", "quantizer_clamp_input", "producer_signed_zero_observer"}
    wrong = copy.deepcopy(golden)
    wrong["outputs"]["Y1"][0][3] = 1
    with pytest.raises(ValueError, match="exact RNE/clamp"):
        verify(cap, wrong)


@pytest.mark.parametrize("spec", [{"quant_scale": 0.3}, {"producer_scale": "print(1)"}, {"dtype": "bf16"}])
def test_quantizer_closed_source_rejects_unmodeled_scalars_or_formats(spec):
    with pytest.raises(ValueError):
        capsule_source.build_loader_src({"op": "producer_quantizer_observer", **spec})


def test_closed_mask_and_composition_effects_follow_selected_actual_source():
    attention = {"op": "attention_residual_norm", "causal": True}
    assert {"causal_mask", "two_contraction_composition", "normalization_composition"} <= source_effects(attention)
    assert "causal_mask" not in source_effects({**attention, "causal": False})
    assert {"multiple_consumers", "residual_composition"} <= source_effects({"op": "mlp_residual"})
    assert {"convolution_source_window", "pooling_composition"} <= source_effects({"op": "conv_residual_pool"})
    assert not source_effects({"op": "unknown"})
