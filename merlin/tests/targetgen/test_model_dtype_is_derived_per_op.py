"""A whole-model compile requests one dtype, but captured operations can be mixed.

This file replaces a ratchet. The ratchet pinned that ``model_op_demands`` stamped the caller's single
``in_fmt`` on every demand, and said why that was survivable *only* for as long as a captured
``linalg.generic`` contraction carried no extent: give it one and a tile WOULD be synthesized, at the
model's declared dtype, which for both captures below is the wrong one for exactly that op. The extent
reader now understands those generics (see ``test_generic_contraction_extents``), so the dtype had to be
derived first, and this states what was derived.

Two facts, not one:

* ``in_fmt`` is what the compile was ASKED for. It stays the caller's single declared format on every
  demand as the requested datapath, but a captured f32 contraction cannot be counted as an int8
  device contraction unless an actual precision transform is present in the program.
* ``elem_fmt`` is what the op's own operands ARE. It is per op, canonical, and UNKNOWN (``None``) when
  it cannot be read -- and ``tile_fmt`` falls back to ``in_fmt`` there, so unknown never widens.

The corpus is mixed because torchAO quantizes Linear weights and leaves Conv2d alone:

* ``M2_microvit_gemmini`` -- 12 ``linalg.matmul`` ops in **i8**, one reducing contraction-tagged
  ``linalg.generic`` in **f32**.
* ``SY_model_resnet50`` -- its current capture has 54 ``linalg.matmul`` contractions in **f32**;
  non-reducing ``linalg.generic`` operations do not count as contractions.
"""

from __future__ import annotations

import pytest

from merlin.common.paths import merlin_dir
from merlin.targetgen.capsule_source import linalg_summary, model_op_demands


def _capsule(name: str) -> str:
    path = next(merlin_dir().joinpath("contract/capsules").rglob(f"{name}/capsule.interface.mlir"), None)
    if path is None:
        pytest.skip(f"{name} is not in this checkout")
    return path.read_text(encoding="utf-8")


def _contraction_formats(text: str) -> dict[str, set[str | None]]:
    """``elem_fmt`` of each contraction demand, grouped by the operation that carries it."""
    summary = linalg_summary(text)
    carriers = [c for c, op in zip(summary["carrying_ops"], summary["prov_ops"]) if op != "fill"]
    out: dict[str, set[str | None]] = {}
    for carrier, demand in zip(carriers, model_op_demands(text, "int8"), strict=True):
        if demand.family == "contraction":
            out.setdefault(carrier, set()).add(demand.elem_fmt)
    return out


@pytest.mark.parametrize(
    "name,named,generic",
    [("M2_microvit_gemmini", "int8", "fp32")],
)
def test_a_mixed_capture_gives_each_contraction_its_own_format(name, named, generic):
    by_carrier = _contraction_formats(_capsule(name))
    assert by_carrier["linalg.matmul"] == {named}
    assert by_carrier["linalg.generic"] == {generic}


def test_resnet_contractions_are_all_captured_in_fp32():
    by_carrier = _contraction_formats(_capsule("SY_model_resnet50"))
    assert by_carrier == {"linalg.matmul": {"fp32"}}


def test_the_declared_format_still_reaches_every_demand():
    """Routing legality is about the format the compile was asked for, so ``in_fmt`` must not follow the
    capture. It used to be the ONLY format on a demand; it is still the one the router reads."""
    text = _capsule("M2_microvit_gemmini")
    for declared in ("int8", "bf16"):
        assert {d.in_fmt for d in model_op_demands(text, declared)} == {declared}


def test_requested_int8_cannot_place_uncorrected_fp32_contractions_on_int8_mesh():
    """Requested precision is not evidence that a captured operation was converted."""
    routing = pytest.importorskip("merlin.targetgen.routing")
    try:
        text = _capsule("M1_lstmnetvit_gemmini")
        demands = model_op_demands(text, "int8")
        mesh = routing.route_plan(demands, "gemmini").get("mesh") or []
    except Exception as exc:  # noqa: BLE001 -- an unresolvable contract is not a verdict
        pytest.skip(f"gemmini contract not resolvable here: {exc}")
    assert not mesh
    assert any(d.elem_fmt == "fp32" and d.in_fmt == "int8" for d in demands)


def test_unconverted_elementwise_uses_its_captured_format_for_legality():
    from merlin.targetgen.compute_units import ComputeUnit, SemanticCapability
    from merlin.targetgen.routing import OpDemand, route

    mesh = ComputeUnit(
        name="array",
        kind="systolic",
        dtypes=("int8",),
        ops=("matmul",),
        semantic_capabilities=(SemanticCapability(family="elementwise_map", dtypes=("int8",)),),
    )
    demand = OpDemand(op="mul", family="elementwise_map", in_fmt="int8", elem_fmt="fp32")
    assert route([demand], [mesh])[0].unit is None


def test_movement_form_does_not_equate_dma_copy_with_standalone_permutation():
    from merlin.targetgen.compute_units import ComputeUnit, SemanticCapability
    from merlin.targetgen.routing import OpDemand, route

    transfer = ComputeUnit(
        name="transfer",
        kind="systolic",
        dtypes=("int8",),
        ops=("copy",),
        semantic_capabilities=(SemanticCapability(family="movement", dtypes=("int8",), forms=("copy",)),),
    )
    permute = OpDemand(op="permute", family="movement", in_fmt="int8", elem_fmt="int8", form="permutation")
    copy = OpDemand(op="copy", family="movement", in_fmt="int8", elem_fmt="int8", form="copy")
    assert route([permute], [transfer])[0].unit is None
    assert route([copy], [transfer])[0].unit == "transfer"
    permuter = ComputeUnit(
        name="permuter",
        kind="vector",
        dtypes=("int8",),
        ops=("copy",),
        semantic_capabilities=(SemanticCapability(family="movement", dtypes=("int8",), forms=("permutation",)),),
    )
    assert route([permute], [permuter])[0].unit == "permuter"


def test_captured_linalg_transpose_is_a_permutation_demand():
    text = (
        "module {\n"
        "  %0 = linalg.transpose ins(%a : tensor<4x8xi8>) outs(%b : tensor<8x4xi8>) "
        'permutation = [1, 0] attrs = {prov.op = "permute", prov.family = "layout"}\n'
        "}\n"
    )
    (demand,) = model_op_demands(text, "int8")
    assert demand.family == "movement"
    assert demand.form == "permutation"


def test_generic_movement_form_classifier_distinguishes_copy_and_permutation():
    from merlin.targetgen.semantic_families import operation_form

    assert operation_form("movement") == "copy"
    assert operation_form("copy") == "copy"
    assert operation_form("permute") == "permutation"
    assert operation_form("", carrier_op="linalg.transpose") == "permutation"
    assert operation_form("gather") is None  # indexed addressing needs separate evidence


def test_an_unreadable_format_keeps_the_declared_one():
    """UNKNOWN never widens. An op whose operands are not one registry format gets ``elem_fmt is None``
    and therefore the declared format for its tile, exactly as before this field existed."""
    text = (
        "module {\n"
        '  %0 = linalg.generic {indexing_maps = [], iterator_types = ["parallel"]} '
        "ins(%a, %b : tensor<4xf32>, tensor<4xi1>) outs(%c : tensor<4xf32>) "
        'attrs = {prov.op = "select", prov.family = "elementwise_map"} { ^bb0: }\n'
        '  %1 = linalg.generic {indexing_maps = [], iterator_types = ["parallel"]} '
        "ins(%d : tensor<4xi128>) outs(%e : tensor<4xi128>) "
        'attrs = {prov.op = "index_shuffle", prov.family = "elementwise_map"} { ^bb0: }\n'
        "}\n"
    )
    demands = {d.op: d for d in model_op_demands(text, "bf16")}
    # operands that disagree with each other have no single format...
    assert demands["select"].elem_fmt is None
    # ...and neither does an element type the format registry does not carry.
    assert demands["index_shuffle"].elem_fmt is None
    for d in demands.values():
        assert d.tile_fmt == "bf16"


def test_the_format_is_canonical_not_the_mlir_spelling():
    """``i8`` in the IR and ``int8`` in a contract are ONE format; resolving through the registry is what
    keeps the tile builder and the router from disagreeing about which."""
    text = (
        "module {\n"
        '  %0 = linalg.matmul {prov.op = "matmul", prov.family = "contraction"} '
        "ins(%a, %b : tensor<8x16xi8>, tensor<16x32xi8>) outs(%c : tensor<8x32xi32>) -> tensor<8x32xi32>\n"
        "}\n"
    )
    (demand,) = model_op_demands(text, "int8")
    assert demand.elem_fmt == "int8" and demand.tile_fmt == "int8"


def test_the_mesh_tile_is_synthesized_in_the_ops_own_format():
    """The consumer half. A tile built at the model's declared format certifies arithmetic the op does
    not perform -- which is only reachable now that such an op carries an extent at all."""
    import inspect

    from merlin.compile import mesh as MESH

    source = inspect.getsource(MESH._mesh_verify)
    assert "_mesh_tile_binding(target, d.tile_fmt, r.acc," in source, (
        "the mesh tile must be bound to the op's own element format (falling back to the declared one "
        "when it is unknown), not to the model-level format every demand carries"
    )


def test_parsed_operand_roles_control_route_and_coverage_not_requested_precision():
    from merlin.common import mlir_query as mq
    from merlin.targetgen import coverage_certificate as cc
    from merlin.targetgen.capsule_source import model_op_demands_checked
    from merlin.targetgen.compute_units import AccumRule, ComputeUnit, SemanticCapability
    from merlin.targetgen.routing import route

    def source(left: str, right: str) -> str:
        return f"""module {{ func.func @forward(%a:tensor<2x3x{left}>, %w:tensor<3x4x{right}>)
            -> tensor<2x4xf32> {{
          %e = tensor.empty() : tensor<2x4xf32>
          %r = linalg.matmul {{prov.op = "matmul", prov.family = "contraction", prov.region_id = "r0"}} ins(%a, %w : tensor<2x3x{left}>, tensor<3x4x{right}>) outs(%e : tensor<2x4xf32>) -> tensor<2x4xf32>
          func.return %r : tensor<2x4xf32>
        }} }}"""

    def unit(name: str, formats: tuple[str, ...], rule: AccumRule) -> ComputeUnit:
        return ComputeUnit(name=name, kind="systolic", dtypes=formats, ops=("matmul",), accumulate=(rule,))

    def certificate(demand, result, formats):
        plan = {
            "results": [result],
            "mesh": [result] if result.unit else [],
            "fallback": [] if result.unit else [result],
            "scalar_rvv": [],
        }
        return cc.build(plan, {"contraction": SemanticCapability(family="contraction", dtypes=formats)})

    mixed_text = source("f32", "i8")
    mq.parse(mixed_text)  # a real parsed source, not a hand-built OpDemand
    (mixed,) = model_op_demands_checked(mixed_text, "int8", "int8")
    assert mixed.captured_input_formats == ("fp32", "int8")
    assert (mixed.in_fmt, mixed.weight_fmt) == ("int8", "int8")  # request retained only as diagnosis
    int8_only = unit("int8", ("int8",), AccumRule("int8", "int8", "i32"))
    assert route([mixed], [int8_only])[0].unit is None
    reversed_roles = unit("reversed", ("fp32", "int8"), AccumRule("int8", "fp32", "f32"))
    assert route([mixed], [reversed_roles])[0].unit is None
    mixed_unit = unit("mixed", ("fp32", "int8"), AccumRule("fp32", "int8", "f32"))
    mixed_route = route([mixed], [mixed_unit])[0]
    assert mixed_route.unit == "mixed" and mixed.tile_fmt == "fp32"
    mixed_cert = certificate(mixed, mixed_route, ("fp32", "int8"))
    assert mixed_cert["n_eligible"] == 1
    assert mixed_cert["regions"][0]["captured_weight_format"] == "int8"

    uniform_text = source("f32", "f32")
    mq.parse(uniform_text)
    (uniform,) = model_op_demands_checked(uniform_text, "int8", "int8")
    assert uniform.captured_input_formats == ("fp32", "fp32")
    fp32_unit = unit("float", ("fp32",), AccumRule("fp32", "fp32", "f32"))
    uniform_route = route([uniform], [fp32_unit])[0]
    assert uniform_route.unit == "float"  # requested int8 weight cannot cause false fallback
    uniform_cert = certificate(uniform, uniform_route, ("fp32",))
    assert uniform_cert["n_eligible"] == 1
    assert uniform_cert["regions"][0]["eligibility_weight_format"] == "fp32"

    partial = mixed_text.replace("tensor<2x3xf32>", "tensor<?x3xf32>")
    parsed_partial = mq.parse(partial)
    (unknown,) = model_op_demands(partial, "int8", "int8")
    assert unknown.captured_input_formats == (None, "int8")
    from merlin.xdsl_dialects._common import text as module_text

    (generic_unknown,) = model_op_demands(module_text(parsed_partial, generic=True), "int8", "int8")
    assert generic_unknown.captured_input_formats == (None, "int8")
    assert not unknown.source_formats_complete and unknown.tile_fmt == "int8"  # compatibility only
    unknown_route = route([unknown], [int8_only])[0]
    assert unknown_route.unit is None
    unknown_cert = certificate(unknown, unknown_route, ("int8",))
    assert unknown_cert["n_eligible"] == 0 and unknown_cert["n_unknown_capture_formats"] == 1

    unrecognized = source("f32", "i1")
    mq.parse(unrecognized)
    (bad_format,) = model_op_demands_checked(unrecognized, "int8", "int8")
    assert bad_format.captured_input_formats == ("fp32", None)
    bad_route = route([bad_format], [mixed_unit])[0]
    assert bad_route.unit is None
    bad_plan = {"results": [bad_route], "mesh": [], "fallback": [bad_route], "scalar_rvv": []}
    incomplete_cert = cc.build(
        bad_plan,
        {"contraction": SemanticCapability(family="contraction", dtypes=("fp32", "int8"))},
        linalg_mlir=unrecognized,
    )
    assert incomplete_cert["n_eligible"] == 0
    assert incomplete_cert["denominator_completeness"]["inventory_status"] == (
        "verified_structure_operand_formats_incomplete"
    )


def test_independent_capability_denominator_preserves_joint_operand_rules():
    from dataclasses import replace

    from merlin.targetgen import coverage_certificate as cc
    from merlin.targetgen.capsule_source import model_op_demands_checked
    from merlin.targetgen.compute_units import AccumRule, ComputeUnit, SemanticCapability, semantic_capability_map
    from merlin.targetgen.routing import route

    text = """module { func.func @forward(%a:tensor<2x3xf32>, %w:tensor<3x4xi8>)
        -> tensor<2x4xf32> {
      %e = tensor.empty() : tensor<2x4xf32>
      %r = linalg.matmul {prov.op = "matmul", prov.family = "contraction", prov.region_id = "r0"} ins(%a, %w : tensor<2x3xf32>, tensor<3x4xi8>) outs(%e : tensor<2x4xf32>) -> tensor<2x4xf32>
      func.return %r : tensor<2x4xf32>
    } }"""
    (demand,) = model_op_demands_checked(text, "int8", "int8")

    def unit(name, fmt, rule):
        return ComputeUnit(
            name=name,
            kind="systolic",
            dtypes=(fmt,),
            ops=("matmul",),
            accumulate=(rule,),
            semantic_capabilities=(SemanticCapability(family="contraction", dtypes=(fmt,)),),
        )

    units = [
        unit("integer", "int8", AccumRule("int8", "int8", "i32")),
        unit("float", "fp32", AccumRule("fp32", "fp32", "f32")),
    ]
    (result,) = route([demand], units)
    assert result.unit is None
    plan = {"results": [result], "mesh": [], "fallback": [result], "scalar_rvv": []}
    capability = semantic_capability_map(units)
    assert set(capability["contraction"].dtypes) == {"int8", "fp32"}
    assert set(capability["contraction"].operand_pairs) == {("int8", "int8"), ("fp32", "fp32")}
    certificate = cc.build(plan, capability, linalg_mlir=text)
    assert certificate["n_eligible"] == 0
    assert certificate["regions"][0]["target_eligible"] is False
    assert certificate["regions"][0]["eligibility_reason"].startswith("input/weight pair")
    from merlin.targetgen import coverage_report
    from merlin.targetgen.eligibility import RegionDescriptor

    assert (
        coverage_report._decline_axis(
            RegionDescriptor(family="contraction", in_dtype="fp32", weight_dtype="int8"),
            "contraction",
            capability,
            undetermined=False,
        )
        == "dtype"
    )

    # A containing unit must not acquire a mixed pair merely by folding its
    # own float capability with an embedded integer unit's capability.
    containing = replace(units[1], contains=("integer",))
    folded = semantic_capability_map([units[0], containing])["contraction"]
    assert set(folded.operand_pairs) == {("int8", "int8"), ("fp32", "fp32")}

    mixed_unit = ComputeUnit(
        name="mixed",
        kind="systolic",
        dtypes=("fp32", "int8"),
        ops=("matmul",),
        accumulate=(AccumRule("fp32", "int8", "f32"),),
        semantic_capabilities=(SemanticCapability(family="contraction", dtypes=("fp32", "int8")),),
    )
    (mixed_result,) = route([demand], [mixed_unit])
    mixed_plan = {"results": [mixed_result], "mesh": [mixed_result], "fallback": [], "scalar_rvv": []}
    assert cc.build(mixed_plan, semantic_capability_map([mixed_unit]), linalg_mlir=text)["n_eligible"] == 1

    # A separately authored semantic pair can state hardware capability wider
    # than the compiler route. That must remain in the denominator as fallback.
    from merlin.targetgen import compute_units as cu
    from merlin.targetgen import eligibility as el

    independent = cu.compute_units(
        {
            "compute_units": [
                {
                    "name": "limited_lowering",
                    "kind": "systolic",
                    "dtypes": ["int8"],
                    "ops": ["matmul"],
                    "accumulate": [{"in": "int8", "weight": "int8", "acc": "i32"}],
                    "semantic_capabilities": [
                        {
                            "family": "contraction",
                            "dtypes": ["int8", "fp32"],
                            "operand_pairs": [{"in": "fp32", "weight": "int8"}],
                        }
                    ],
                }
            ]
        }
    )
    assert route([demand], independent)[0].unit is None
    assert cc.build(plan, semantic_capability_map(independent), linalg_mlir=text)["n_eligible"] == 1
    # Direct legacy descriptors never supplied joint support. Preserve that API
    # explicitly; it is not a proof for a projected, captured model source.
    assert el.is_eligible(
        el.RegionDescriptor(family="contraction", in_dtype="fp32", weight_dtype="int8"),
        {"contraction": SemanticCapability(family="contraction", dtypes=("fp32", "int8"))},
    ).eligible
