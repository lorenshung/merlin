"""Profitability follows geometry, source work and retained host epilogues."""

import copy

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen.placement_cost import estimate, host_work

RATES = dict(
    host_word_bits=64,
    host_arithmetic_cycles=1,
    host_math_cycles=32,
    host_index_cycles=1,
    host_copy_cycles_per_byte=0.125,
    host_quantize_cycles_per_byte=8,
    dma_cycles_per_byte=0.0625,
    instruction_issue_cycles=1,
    compute_tile_cycles=16,
    fence_cycles=32,
    dispatch_cycles=64,
)
DECLARATION = dict(schema="merlin.placement_cost.v1", rates=RATES)
FACTS = dict(facts=dict(arrays=[dict(rows=16, cols=16, corroborated=True)]))


def source(size):
    return parse_mlir_text(f"""builtin.module {{
      func.func @forward(%a: tensor<{size}xf32>) -> tensor<{size}xf32> {{
        %e = tensor.empty() : tensor<{size}xf32>
        %v = linalg.generic {{indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],
                             iterator_types=["parallel"]}}
          ins(%a : tensor<{size}xf32>) outs(%e : tensor<{size}xf32>) {{
          ^bb0(%x: f32, %unused: f32):
            %n = arith.negf %x : f32
            linalg.yield %n : f32
        }} -> tensor<{size}xf32>
        func.return %v : tensor<{size}xf32>
      }}
    }}""")


def plan(module):
    return dict(
        sites=[
            dict(
                status="admitted",
                source_operation_ordinal=next(i for i, op in enumerate(module.walk()) if op.name == "linalg.generic"),
            )
        ]
    )


def test_tiny_work_is_unprofitable_and_cost_increases_with_traffic():
    small, big = source(1), source(4096)
    a = estimate(small, small, plan(small), DECLARATION, FACTS)
    b = estimate(big, big, plan(big), DECLARATION, FACTS)
    assert a["reason_code"] == "device_cost_not_profitable"
    assert b["transfer_bytes"] == 4096 * a["transfer_bytes"]
    assert b["device_parts"]["dma"] > a["device_parts"]["dma"]
    assert a["device_parts"]["fences"] == b["device_parts"]["fences"] == 32


def test_actual_geometry_and_host_epilogue_are_charged():
    module = source(4096)
    ordinary = estimate(module, module, plan(module), DECLARATION, FACTS)
    bigger = dict(facts=dict(arrays=[dict(rows=32, cols=32, corroborated=True)]))
    large = estimate(module, module, plan(module), DECLARATION, bigger)
    assert large["device_parts"]["compute"] < ordinary["device_parts"]["compute"]
    assert ordinary["device_parts"]["host_pre_post"] == 0
    assert host_work(module, RATES) == 4096


def test_unknown_facts_never_create_a_profitable_placement():
    module = source(4096)
    result = estimate(module, module, plan(module), DECLARATION, {})
    assert result["status"] == "host" and result["reason_code"] == "placement_cost_unproved"


def test_authored_rates_are_the_only_timing_authority():
    module = source(4096)
    declaration = copy.deepcopy(DECLARATION)
    declaration["rates"]["dma_cycles_per_byte"] = 100
    cost = estimate(module, module, plan(module), declaration, FACTS)
    assert cost["status"] == "host"
    assert cost["device_parts"]["dma"] == 2 * 4096 * 4 * 100


def test_unprofitable_candidate_restores_original_host_boundary(tmp_path):
    import json

    import yaml

    from merlin.targetgen.core_aten_profitability import prepare

    original = str(source(1))
    (tmp_path / "model.mlir").write_text(original)
    (tmp_path / "inputs.npz").write_bytes(b"original input bytes")
    (tmp_path / "semantic_io.json").write_text("original ABI")
    specification = tmp_path / "software.yaml"
    specification.write_text(yaml.safe_dump(dict(placement_cost_model=DECLARATION)))
    report = dict(cases=[dict(case_id="independent", input_abi=[dict(shape=[1])])])
    before = copy.deepcopy(report)

    def preparation(directory, report, **kwargs):
        module = source(1)
        (directory / "model.mlir").write_text(str(module))
        (directory / "inputs.npz").write_bytes(b"quantized input bytes")
        (directory / "semantic_io.json").write_text("different ABI")
        report["cases"][0]["input_abi"] = [dict(shape=[1, 1])]
        return dict(plan(module), selection=dict(identities={}), exact_dtype_sites=1)

    decision = prepare(tmp_path, report, preparation, software_spec=specification, facts=FACTS)
    assert decision["refusal_code"] == "device_cost_not_profitable"
    assert decision["exact_dtype_sites"] == 0
    assert (tmp_path / "model.mlir").read_text() == original
    assert (tmp_path / "inputs.npz").read_bytes() == b"original input bytes"
    assert (tmp_path / "semantic_io.json").read_text() == "original ABI"
    assert report["cases"][0]["input_abi"] == before["cases"][0]["input_abi"]
    stored = json.loads((tmp_path / "spec-int8-admission.json").read_text())
    assert stored["placement_cost"]["status"] == "host"
