"""Explicit external catalog calls retain ABI, source order and complete graph accounting."""

from dataclasses import replace

import pytest
from xdsl.dialects.builtin import StringAttr
from xdsl.utils.exceptions import VerifyException

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.xdsl_dialects.lowering.dispatch_program import (
    build_dispatch_program,
    lower_model_to_dispatch_program,
)
from merlin.xdsl_dialects.lowering.global_plan import ValueRepresentation
from merlin.xdsl_dialects.lowering.global_plan_emission import emit_global_plan
from merlin.xdsl_dialects.lowering.outline import OutlineError, outline_dispatches
from merlin.xdsl_dialects.lowering.outlined_plan_emission import (
    OutlinedGlobalPlanEmitter,
    _expanded_driver,
    plan_dispatch_fusion,
)

SOURCE = """
builtin.module {
  func.func private @external(tensor<2x2xf32>) -> tensor<2x2xf32>
      attributes {compiler.contract = "original"}
  func.func private @notify(tensor<2x2xf32>)
  func.func @forward(%x: tensor<2x2xf32>, %w: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %q = func.call @external(%x) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %e = tensor.empty() : tensor<2x2xf32>
    %c = arith.constant 0.0 : f32
    %f = linalg.fill ins(%c : f32) outs(%e : tensor<2x2xf32>) -> tensor<2x2xf32>
    %y = linalg.matmul ins(%q, %w : tensor<2x2xf32>, tensor<2x2xf32>)
        outs(%f : tensor<2x2xf32>) -> tensor<2x2xf32>
    func.call @notify(%y) : (tensor<2x2xf32>) -> ()
    %z = func.call @external(%y) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    func.return %z : tensor<2x2xf32>
  }
}
"""


def _capture(*, prune=False):
    return lower_model_to_dispatch_program(
        parse_mlir_text(SOURCE), prune=prune, external_symbols=("external", "notify")
    )


def _plan(graph, *, fuse):
    return plan_dispatch_fusion(
        graph,
        [tuple(range(len(graph.nodes)))] if fuse else [],
        placement="compiler_function",
        representation=lambda name: ValueRepresentation("tensor_ssa", "logical", graph.buffers[name].dtype),
    )


def test_catalog_declarations_need_explicit_permission():
    with pytest.raises(OutlineError, match="@external"):
        outline_dispatches(parse_mlir_text(SOURCE))
    with pytest.raises(OutlineError, match="@notify"):
        outline_dispatches(parse_mlir_text(SOURCE), external_symbols=("external",))


@pytest.mark.parametrize("symbols", [("missing",), ("forward",), ("external", "external"), "external", (0,)])
def test_permission_cannot_invent_a_declaration_or_authorize_a_definition(symbols):
    with pytest.raises(OutlineError, match="external"):
        outline_dispatches(parse_mlir_text(SOURCE), external_symbols=symbols)


@pytest.mark.parametrize("prune", [False, True])
def test_mixed_catalog_and_outlined_calls_keep_source_order_and_effect_only_calls(prune):
    outlined, graph = _capture(prune=prune)
    assert outlined.n_kernels == 1
    assert outlined.external_symbols == ("external", "notify")
    calls = [node for node in graph.nodes if node.kind == "dispatch"]
    assert [node.op for node in calls] == ["external", "forward$kernel_0", "notify", "external"]
    assert not calls[2].outputs
    assert calls[1].inputs[0] == calls[0].outputs[0]
    assert calls[2].inputs == calls[1].outputs == calls[3].inputs
    outlined.module.verify()


@pytest.mark.parametrize("fuse", [False, True])
def test_whole_graph_emission_preserves_declared_calls_without_claiming_object_closure(fuse):
    outlined, graph = _capture()
    before = str(outlined.module)
    plan = _plan(graph, fuse=fuse)
    emitter = OutlinedGlobalPlanEmitter(outlined)
    emission = emit_global_plan(graph, plan, emitter)
    assert str(outlined.module) == before
    emitter.module.verify()
    assert emission.dispatch.results == graph.results
    assert emitter.proof["external_declarations"] == ["external", "notify"]
    assert emitter.proof["external_implementation_closure"] == "UNKNOWN"
    assert emitter.proof["timing"] == "UNKNOWN"
    assert emitter.proof["computation"] == "expanded_driver_structurally_equivalent"
    external = next(
        op for op in emitter.module.body.block.ops if op.name == "func.func" and op.sym_name.data == "external"
    )
    assert not external.body.blocks
    assert external.attributes["compiler.contract"] == StringAttr("original")


def test_dispatch_table_refuses_same_count_but_wrong_symbol():
    outlined, _ = _capture()
    outlined.dispatches[0] = replace(outlined.dispatches[0], symbol="different_kernel")
    with pytest.raises(OutlineError, match="disagrees with driver call"):
        build_dispatch_program(outlined)


def test_external_call_signature_must_match_the_declaration():
    malformed = SOURCE.replace(
        "@external(tensor<2x2xf32>) -> tensor<2x2xf32>", "@external(tensor<2x2xi32>) -> tensor<2x2xf32>"
    )
    with pytest.raises(VerifyException, match="type|Type"):
        outline_dispatches(parse_mlir_text(malformed), external_symbols=("external", "notify"))


def test_expansion_proof_includes_external_declaration_attributes():
    outlined, _ = _capture()
    changed = outlined.module.clone()
    declaration = next(op for op in changed.body.block.ops if op.name == "func.func" and op.sym_name.data == "external")
    declaration.attributes["compiler.contract"] = StringAttr("changed")
    assert not _expanded_driver(outlined.module, "forward", outlined.external_symbols).is_structurally_equivalent(
        _expanded_driver(changed, "forward", outlined.external_symbols)
    )
