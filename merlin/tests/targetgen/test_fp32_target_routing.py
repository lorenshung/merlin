"""A second datapath routes from selected facts without changing shared target policy."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.common.paths import repo_root
from merlin.system.offload import facts_dtype_triples
from merlin.targetgen import core_aten_device as device
from merlin.targetgen.plugins import load_module

pytestmark = pytest.mark.target("gemmini_fp32")

SOURCE = """builtin.module {
  func.func @forward(%a: tensor<2x3xf32>, %b: tensor<3x4xf32>) -> tensor<2x4xf32> {
    %z = arith.constant 0.0 : f32
    %init = tensor.splat %z : tensor<2x4xf32>
    %r = linalg.matmul ins(%a, %b : tensor<2x3xf32>, tensor<3x4xf32>) outs(%init : tensor<2x4xf32>) -> tensor<2x4xf32>
    func.return %r : tensor<2x4xf32>
  }
}"""


def facts(operand, accum, weight=None):
    paths = [{"name": "input", "dtype": operand}, {"name": "accumulator", "dtype": accum}]
    if weight:
        paths.append({"name": "weight", "dtype": weight})
    return {"facts": {"target": "other_device", "datapaths": paths}}


def test_precision_triples_preserve_weight_and_fail_closed():
    assert facts_dtype_triples(facts("f32", "f32")) == (("f32", "f32", "f32"),)
    assert facts_dtype_triples(facts("i8", "i32")) == (("i8", "i8", "i32"),)
    assert facts_dtype_triples(facts("i8", "i32", "f16")) == (("i8", "f16", "i32"),)
    assert facts_dtype_triples(facts("unknown", "f32")) == ()
    assert facts_dtype_triples({"facts": {}}) == ()


def test_source_bound_routing_uses_supplied_facts_for_f32(tmp_path, monkeypatch):
    (tmp_path / "model.mlir").write_text(SOURCE)
    package = tmp_path / "submission"
    package.mkdir()
    seen = []
    backend = SimpleNamespace(
        build_catalog=lambda source: (
            seen.append(hashlib.sha256(source.encode()).hexdigest()),
            {"covered_contractions": 1},
        ),
        merlin_builder=lambda llvm: lambda *args: None,
    )
    monkeypatch.setattr(device, "load_module", lambda *a, **k: backend)
    monkeypatch.setattr("merlin.llvmlower.toolchain.llvm_install", lambda: Path("/selected/llvm"))
    monkeypatch.setattr("merlin.system.offload.device_contraction_ranks", lambda target: None)
    provider = SimpleNamespace(routing=device.submitted_catalog_routing)
    route, _ = device.routing_for_bundle(tmp_path, "other_device", package, provider, facts("f32", "f32"))
    assert route.device == "other_device" and route.operand_dtype == route.accum_dtype == "f32"
    assert route.catalog_builder is not None and route.select is None
    assert seen == [hashlib.sha256(SOURCE.encode()).hexdigest()]
    route, why = device.routing_for_bundle(tmp_path, "other_device", package, provider, facts("i8", "i32"))
    assert route is None and "dtypes" in why


def test_fp32_descriptor_and_isolated_extension_selection(tmp_path, monkeypatch):
    from merlin.targetgen.target_experiment import load_target_experiment
    from merlin.targetgen.target_registry import resolve

    root = repo_root() / "examples/gemmini_fp32"
    descriptor = load_target_experiment(root / "target/descriptor.yaml")
    assert descriptor.target == "gemmini_fp32"
    assert descriptor.host_board == "gemmini_fp32_rocket"
    assert descriptor.selected_board_catalog() == root / "target/board-catalog.yaml"
    assert resolve(descriptor.target).contract_path == root / "target/contracts/target_contract.yaml"
    provider = load_module(root / "phase0/core_aten", "execution_provider.py", package_name="fp32_provider_test")
    library = tmp_path / "isolated/libgemmini.so"
    library.parent.mkdir()
    library.write_bytes(b"isolated extension fixture")
    monkeypatch.setenv("MERLIN_FP32_SPIKE_EXTLIB", str(library))
    monkeypatch.setattr("merlin.runtime.backends.spike.spike_path", lambda: "/selected/tools/bin/spike")
    options = provider.runner_options()
    assert options["extlib"] == library and options["extension"] == "gemmini"
    library.unlink()
    with pytest.raises(ValueError, match="isolated FP32"):
        provider.runner_options()


PRODUCT_REDUCTION = """builtin.module {
  func.func @forward(%x: tensor<5x7xf32>, %y: tensor<5x7xf32>) -> tensor<5xf32> {
    %z = arith.constant 0.0 : f32
    %init = tensor.splat %z : tensor<5xf32>
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%x, %y : tensor<5x7xf32>, tensor<5x7xf32>) outs(%init : tensor<5xf32>) {
    ^bb0(%a: f32, %b: f32, %acc: f32):
      %diff = arith.subf %a, %b : f32
      %product = arith.mulf %diff, %diff : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<5xf32>
    func.return %r : tensor<5xf32>
  }
}"""


def test_product_reduction_is_a_demand_without_dense_rewrite_permission():
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.kernels.shapes import observe_contraction_demands, observe_contractions

    module = parse_mlir_text(PRODUCT_REDUCTION)
    assert observe_contractions(module) == []
    demands = observe_contraction_demands(module)
    assert len(demands) == 1
    shape = demands[0][1]
    assert shape.parallel == (5,) and shape.reduction == (7,) and shape.dtypes == ("f32", "f32", "f32")


@pytest.mark.parametrize(
    "changed",
    [
        PRODUCT_REDUCTION.replace("arith.mulf %diff, %diff", "arith.addf %diff, %diff"),
        PRODUCT_REDUCTION.replace("arith.mulf %diff, %diff", "arith.mulf %acc, %diff"),
        PRODUCT_REDUCTION.replace("linalg.yield %sum", "linalg.yield %product"),
    ],
)
def test_nonproduct_and_accumulator_dependent_reductions_are_not_demands(changed):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.kernels.shapes import observe_contraction_demands

    assert observe_contraction_demands(parse_mlir_text(changed)) == []


def test_selected_193_capture_placement():
    import json
    import os

    from merlin_experiments.phase0.core_aten_stage import contraction_lane

    capture_root = os.environ.get("MERLIN_CORE_ATEN_CAPTURES")
    facts_path = os.environ.get("MERLIN_RTL_FACTS")
    if not capture_root or not facts_path:
        pytest.skip("requires explicitly selected canonical captures and extracted FP32 facts")
    document = json.loads(Path(facts_path).read_bytes())
    assert document["facts"]["target"] == "gemmini_fp32"
    assert facts_dtype_triples(document) == (("f32", "f32", "f32"),)
    captures = sorted(Path(capture_root).glob("*/capsule.linalg.mlir"))
    assert len(captures) == 193
    devices = {
        json.loads((source.parent / "capture.json").read_bytes())["overload"]
        for source in captures
        if contraction_lane(source, document) == "device"
    }
    assert devices == {
        "aten.mm.default",
        "aten.addmm.default",
        "aten.bmm.default",
        "aten.convolution.default",
        "aten.convolution_backward.default",
        "aten._cdist_forward.default",
        "aten._pdist_forward.default",
        "aten.upsample_bilinear2d.vec",
        "aten.upsample_nearest2d.vec",
        "aten._fft_r2c.default",
        "aten._fft_c2r.default",
    }
    assert len(captures) - len(devices) == 182


def test_foreign_facts_are_refused_before_execution(tmp_path, monkeypatch):
    import json
    import os

    selected = tmp_path / "facts.json"
    selected.write_text(json.dumps(facts("f32", "f32")))
    monkeypatch.setenv("MERLIN_RTL_FACTS", "previous-selection")
    with pytest.raises(ValueError, match="different target"):
        with device.selected_facts("gemmini_fp32", selected):
            pytest.fail("foreign facts must not be attributed to this target")
    assert os.environ["MERLIN_RTL_FACTS"] == "previous-selection"


def test_f32_dense_rewrite_derives_precision_from_facts():
    from merlin.common import mlir_query as mq
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.device_offload import rewrite_contractions_to_device
    from merlin.targetgen.rtl.facts import observed_facts
    from merlin.targetgen.target_registry import observed_contract

    contract = {"name": "gemmini_fp32", "endpoint_kind": "inline_asm_insn", "compute_units": []}
    document = facts("f32", "f32")
    document["facts"]["target"] = "gemmini_fp32"
    with observed_contract("gemmini_fp32", contract), observed_facts("gemmini_fp32", document):
        module = parse_mlir_text(SOURCE)
        rewritten = rewrite_contractions_to_device(module, "gemmini_fp32", select=lambda _: True)
    assert rewritten.moved == 1
    assert rewritten.routed[0].dtypes == ("f32", "f32", "f32")
    assert mq.op_count(module, "linalg.matmul") == 0 and mq.op_count(module, "func.call") == 1
