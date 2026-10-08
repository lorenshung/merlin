"""Promised composition and immutable source/emission closure on the normal hook."""

from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.llvmlower.host_transform_chain import (
    FILENAME,
    HostLLVMArtifact,
    HostLLVMEmission,
    HostLLVMStageContract,
    HostLLVMTransformChain,
    HostLLVMTransformStage,
    recheck_host_transform_chain,
)
from merlin.llvmlower.lowering_recipe import LoweringRecipe, bind_host_transform_chain
from merlin.runtime.backends.spike_model import _transform_host_ir


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.ll"
    path.write_text("define i32 @value(i32 %x) { ret i32 %x }\n")
    return path


def _stage(tmp_path, name, *, transform=None, verify=None):
    proof = tmp_path / (name + "_permission.json")
    proof.write_text('{"source_arithmetic_unchanged":true}\n')

    def copy(original, work):
        selected = work / "selected.ll"
        selected.write_bytes(original.read_bytes())
        return selected

    def verify_copy(original, selected, work):
        if original.read_bytes() != selected.read_bytes():
            raise ValueError("copy stage changed the complete original IR")
        evidence = work / "emission.json"
        evidence.write_text(json.dumps({"complete_IR_bytes_identical": True}))
        return HostLLVMEmission(
            HostLLVMArtifact.capture(original),
            HostLLVMArtifact.capture(selected),
            (HostLLVMArtifact.capture(evidence),),
        )

    transform, verify = transform or copy, verify or verify_copy
    sources = tuple(
        HostLLVMArtifact.capture(path)
        for path in sorted({Path(inspect.getsourcefile(callback)).resolve() for callback in (transform, verify)})
    )
    contract = HostLLVMStageContract(name, sources, (HostLLVMArtifact.capture(proof),))
    return HostLLVMTransformStage(contract, transform, verify)


def _chain(*stages):
    return HostLLVMTransformChain(tuple(stage.contract for stage in stages), tuple(stages))


def test_added_stages_consume_each_other_and_cannot_drop_selected_terminal(source, tmp_path):
    first, second = _stage(tmp_path, "first"), _stage(tmp_path, "second")
    observed = []

    def terminal(original, work):
        observed.append(original)
        assert original.parent.name == "stage_1_second"
        return original

    selected, receipt = _transform_host_ir(source, tmp_path / "chain", terminal, chain=_chain(first, second))
    record = recheck_host_transform_chain(Path(receipt["chain"]["path"]))
    assert len(observed) == 1 and selected == observed[0]
    assert [row["contract"]["name"] for row in record["stages"]] == ["first", "second"]
    assert record["legacy_terminal"]["input"] == record["stages"][-1]["output"]
    assert source.read_bytes() == selected.read_bytes()


@pytest.mark.parametrize("mutation", ["missing", "order", "duplicate", "untyped", "mutable"])
def test_promised_chain_refuses_incomplete_selection_before_invocation(source, tmp_path, mutation):
    first, second = _stage(tmp_path, "first"), _stage(tmp_path, "second")
    chain = _chain(first, second)
    if mutation == "missing":
        chain = replace(chain, stages=(first,))
    elif mutation == "order":
        chain = replace(chain, stages=(second, first))
    elif mutation == "duplicate":
        chain = _chain(first, first)
    elif mutation == "untyped":
        chain = replace(chain, promised_stages=({},))
    else:
        chain = replace(chain, stages=[first, second])
    work = tmp_path / "refused"
    with pytest.raises((ValueError, TypeError)):
        _transform_host_ir(source, work, None, chain=chain)
    assert not list(work.glob("stage_*"))
    assert json.loads((work / FILENAME).read_text())["status"] == "refused"


def test_normal_build_refuses_a_missing_promised_stage_before_tool_or_lowering_use(source, tmp_path, monkeypatch):
    from merlin.runtime.backends import spike_model

    stage = _stage(tmp_path, "required")
    chain = replace(_chain(stage), stages=())
    monkeypatch.setattr(spike_model, "lower_model_file", lambda *_a, **_k: pytest.fail("lowering must not execute"))
    with pytest.raises(ValueError, match="complete promised order"):
        spike_model.build(source.parent, tmp_path / "build", host_llvm_transform_chain=chain)
    assert not (tmp_path / "build/model.o").exists()


@pytest.mark.parametrize(
    "mutation", ["semantic_proof", "callback_owner", "source_mutation", "untyped_emission", "false_output"]
)
def test_stage_source_proof_and_actual_emission_must_close(source, tmp_path, mutation):
    def transform(original, work):
        if mutation == "source_mutation":
            original.write_text("modified")
        selected = work / "selected.ll"
        selected.write_bytes(original.read_bytes())
        return selected

    stage = _stage(tmp_path, "selected", transform=transform)
    if mutation == "semantic_proof":
        stage.contract.semantic_proofs[0].path.write_text("different permission")
    elif mutation == "callback_owner":
        stage = replace(stage, contract=replace(stage.contract, sources=stage.contract.semantic_proofs))
    elif mutation == "untyped_emission":
        stage = replace(stage, verify_emission=lambda *_: True)
    elif mutation == "false_output":
        original_verify = stage.verify_emission

        def wrong_output(original, selected, work):
            emission = original_verify(original, selected, work)
            return replace(emission, selected=replace(emission.selected, sha256="0" * 64))

        stage = replace(stage, verify_emission=wrong_output)
    before = source.read_bytes()
    with pytest.raises((ValueError, TypeError)):
        _transform_host_ir(source, tmp_path / "refused", None, chain=_chain(stage))
    assert source.read_bytes() == before
    assert json.loads((tmp_path / "refused" / FILENAME).read_text())["status"] == "refused"


def test_later_mutation_and_post_completion_drift_cannot_retain_success(source, tmp_path):
    stage = _stage(tmp_path, "copy")

    def terminal(original, work):
        stage.contract.semantic_proofs[0].path.write_text("changed after stage verification")
        return original

    with pytest.raises(ValueError, match="identity changed"):
        _transform_host_ir(source, tmp_path / "mutated", terminal, chain=_chain(stage))
    assert json.loads((tmp_path / "mutated" / FILENAME).read_text())["status"] == "refused"

    fresh = _stage(tmp_path, "fresh")
    selected, receipt = _transform_host_ir(source, tmp_path / "completed", None, chain=_chain(fresh))
    selected.write_text("changed after emission")
    with pytest.raises(ValueError, match="identity changed"):
        recheck_host_transform_chain(Path(receipt["chain"]["path"]))


def test_reused_workdir_cannot_expose_prior_completed_chain(source, tmp_path):
    stage, work = _stage(tmp_path, "copy"), tmp_path / "chain"
    _transform_host_ir(source, work, None, chain=_chain(stage))
    with pytest.raises(ValueError, match="fresh owned"):
        _transform_host_ir(source, work, None, chain=_chain(stage))
    assert json.loads((work / FILENAME).read_text())["status"] == "refused"


def test_actual_chain_extends_upstream_receipt_without_replacing_original_returned_identity(source, tmp_path):
    stage = _stage(tmp_path, "copy")
    recipe = LoweringRecipe(tmp_path, features=(), sources={"input": source})
    recipe.returned(source.read_text())
    selected, receipt = _transform_host_ir(source, tmp_path / "chain", None, chain=_chain(stage))
    chain_receipt = Path(receipt["chain"]["path"])
    before = json.loads(recipe.path.read_text())["returned_llvm_ir"]
    bind_host_transform_chain(recipe.path, chain_receipt, source=source, selected=selected)
    record = json.loads(recipe.path.read_text())
    assert record["returned_llvm_ir"] == before
    assert record["selected_host_llvm_ir"] == HostLLVMArtifact.capture(selected).to_dict()
    assert record["host_transform_chain"] == HostLLVMArtifact.capture(chain_receipt).to_dict()
    wrong = tmp_path / "different.ll"
    wrong.write_text(source.read_text() + "; different source\n")
    with pytest.raises(ValueError, match="actual returned"):
        bind_host_transform_chain(recipe.path, chain_receipt, source=wrong, selected=selected)


def test_normal_upstream_and_object_build_bind_the_selected_complete_chain(tmp_path):
    import numpy as np

    from merlin.llvmlower import toolchain
    from merlin.runtime.backends import spike, spike_model

    if not all(path.is_file() for path in (toolchain.clang(), toolchain.m2m_python(), spike.gcc_path())):
        pytest.skip("selected upstream and cross compiler unavailable")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "model.mlir").write_text("""module {
      func.func @forward(%a: tensor<3xf32>) -> tensor<3xf32> {
        %init = tensor.empty() : tensor<3xf32>
        %out = linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"]
        } ins(%a : tensor<3xf32>) outs(%init : tensor<3xf32>) {
          ^bb0(%x: f32, %unused: f32):
            %two = arith.constant 2.0 : f32
            %answer = arith.addf %x, %two : f32
            linalg.yield %answer : f32
        } -> tensor<3xf32>
        return %out : tensor<3xf32>
      }
    }""")
    (bundle / "weights.safetensors.manifest.json").write_text('{"0":{"kind":"input","name":"values"}}')
    np.savez(bundle / "inputs.npz", in0=np.array([-3, 0, 7], dtype=np.float32))
    stage = _stage(tmp_path, "checked_copy")

    def terminal(original, work):
        selected = work / "terminal.ll"
        selected.write_bytes(original.read_bytes())
        return selected

    work = tmp_path / "normal"
    result = spike_model.build(
        bundle,
        work,
        backend="scalar",
        host_vectorize=False,
        arena_mb=1,
        host_llvm_transform=terminal,
        host_llvm_transform_chain=_chain(stage),
    )
    host = result["host_llvm_transform"]
    chain = recheck_host_transform_chain(Path(host["chain"]["path"]))
    lowering = json.loads((work / "lower/lowering_recipe.json").read_text())
    compilation = json.loads((work / "compilation_recipe.json").read_text())
    assert lowering["host_transform_chain"] == host["chain"]
    assert lowering["selected_host_llvm_ir"] == chain["selected"]
    assert compilation["preparation"]["host_transform_chain"] == host["chain"]
    model_command = next(row for row in compilation["commands"] if Path(row["requested_output"]).name == "model.o")
    assert model_command["inputs"] == [chain["selected"]]
    assert compilation["status"] == "completed" and result["elf"].is_file()


@pytest.mark.parametrize("mutation", ["missing_stage", "different_order", "no_emission", "strip_promise_and_stage"])
def test_recheck_refuses_receipt_with_an_unfulfilled_promise(source, tmp_path, mutation):
    stages = _stage(tmp_path, "first"), _stage(tmp_path, "second")
    chain = _chain(*stages)
    _, result = _transform_host_ir(source, tmp_path / "chain", None, chain=chain)
    path = Path(result["chain"]["path"])
    record = json.loads(path.read_text())
    if mutation == "missing_stage":
        record["stages"].pop()
    elif mutation == "different_order":
        record["stages"].reverse()
    elif mutation == "strip_promise_and_stage":
        record["promised_order"].pop()
        record["stages"].pop()
    else:
        record["stages"][0]["evidence"] = []
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="promised|emission"):
        recheck_host_transform_chain(path, expected_chain=chain)
