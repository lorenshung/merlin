"""Reference-scale policies retain independent exactness and per-shape gates."""

import pytest
from test_placement_cost import source

from merlin.common.paths import repo_root
from merlin.common.yaml import load_yaml
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen.placement_cost import estimate
from merlin.targetgen.placement_reference import estimate_reference, movement_pattern


def inputs():
    declaration = load_yaml(repo_root() / "examples/gemmini/target/software-spec.yaml")["placement_cost_model"]
    facts = dict(
        facts=dict(
            arrays=[dict(rows=16, cols=16, corroborated=True)],
            memories=[dict(name="scratchpad", row_elems=16, elem_bits=8)],
        )
    )
    return declaration, facts


def candidate(module, family="elementwise_map", numerical="F2"):
    ordinal = next(i for i, op in enumerate(module.walk()) if op.name == "linalg.generic")
    return dict(
        numerical_family=numerical, sites=[dict(status="admitted", family=family, source_operation_ordinal=ordinal)]
    )


def test_reference_cost_is_identical_at_independent_vector_sizes():
    declaration, facts = inputs()
    a, b = source(7), source(8191)
    assert estimate_reference(a, candidate(a), declaration, facts) == estimate_reference(
        b, candidate(b), declaration, facts
    )
    assert (
        estimate(a, a, candidate(a), declaration, facts)["device_cycles"]
        != estimate(b, b, candidate(b), declaration, facts)["device_cycles"]
    )
    assert estimate_reference(a, candidate(a), declaration, facts)["status"] == "device"


def test_scalar_work_is_still_host():
    declaration, facts = inputs()
    module = source(1)
    assert (
        estimate_reference(module, candidate(module), declaration, facts)["reason_code"] == "scalar_work_retained_host"
    )


def test_rates_can_evict_a_reference_family_without_roster_exceptions():
    declaration, facts = inputs()
    declaration["reference_scale"]["rates"]["dma_bytes_per_cycle"] = 0.001
    module = source(6)
    result = estimate_reference(module, candidate(module), declaration, facts)
    assert result["status"] == "host" and result["reason_code"] == "reference_family_not_profitable"


def movement(reverse=False):
    module = source(7)
    from xdsl.rewriter import Rewriter

    negation = next(op for op in module.walk() if op.name == "arith.negf")
    Rewriter.replace_op(negation, [], [negation.operands[0]])
    if reverse:
        from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr
        from xdsl.ir.affine import AffineDimExpr, AffineMap

        op = next(op for op in module.walk() if op.name == "linalg.generic")
        op.properties["indexing_maps"] = ArrayAttr(
            [AffineMapAttr(AffineMap(1, 0, (-AffineDimExpr(0) + 6,))), AffineMapAttr(AffineMap.identity(1))]
        )
    return module


def test_row_selector_and_reversal_have_different_reference_costs():
    declaration, facts = inputs()
    row, gather = movement(), movement(True)
    a = estimate_reference(row, candidate(row, "movement", "F1"), declaration, facts)
    b = estimate_reference(gather, candidate(gather, "movement", "F1"), declaration, facts)
    assert a["status"] == "device" and a["pattern"] == "row_contiguous"
    assert b["status"] == "host" and b["pattern"] == "element_gather"
    assert b["selector_scalar_operations"] == 2
    assert b["device_cycles"] > a["device_cycles"]


def test_strided_merge_accounts_for_bounds_congruence_and_selection():
    module = parse_mlir_text("""module {
      func.func @forward(%x: tensor<3x7xf32>, %y: tensor<3x3xf32>) -> tensor<3x7xf32> {
        %r = "tensor.insert_slice"(%y, %x) <{
          static_offsets=array<i64: 0, 1>, static_sizes=array<i64: 3, 3>, static_strides=array<i64: 1, 2>,
          operandSegmentSizes=array<i32: 1, 1, 0, 0, 0>}>
          : (tensor<3x3xf32>, tensor<3x7xf32>) -> tensor<3x7xf32>
        func.return %r : tensor<3x7xf32>
      }}""")
    assert movement_pattern(module)[0] == "element_gather"
    assert movement_pattern(module)[2] == 6


@pytest.mark.parametrize("value", [0, -1, float("inf")])
def test_invalid_reference_rates_refuse(value):
    declaration, facts = inputs()
    declaration["reference_scale"]["rates"]["device_mac_per_cycle"] = value
    module = source(6)
    assert (
        estimate_reference(module, candidate(module), declaration, facts)["reason_code"]
        == "reference_family_cost_unproved"
    )


@pytest.mark.parametrize("policy", ["reference_scale", "strict_per_shape"])
def test_selectable_policy_records_both_verdicts(tmp_path, monkeypatch, policy):
    import json

    import yaml

    from merlin.targetgen.core_aten_profitability import prepare

    declaration, facts = inputs()
    module = source(6)
    (tmp_path / "model.mlir").write_text(str(module))
    spec = tmp_path / "software.yaml"
    spec.write_text(yaml.safe_dump(dict(placement_cost_model=declaration)))
    report = dict(cases=[dict(case_id="independent", status="bundled")])
    monkeypatch.setenv("MERLIN_PLACEMENT_POLICY", policy)

    def implementation(directory, report, **kwargs):
        return dict(candidate(module), exact_dtype_sites=1, selection=dict(identities={}))

    plan = prepare(tmp_path, report, implementation, software_spec=spec, facts=facts)
    cost = plan["placement_cost"]
    assert cost["policy"] == policy
    assert cost["strict_policy"]["status"] == "host"
    assert cost["status"] == ("device" if policy == "reference_scale" else "host")
    stored = json.loads((tmp_path / "spec-int8-admission.json").read_text())
    assert stored["placement_cost"] == cost
    assert report["cases"][0]["numeric_admission"]["placement_cost"] == cost


def test_no_numeric_candidate_has_explicit_not_applicable_strict_evidence(tmp_path):
    import yaml

    from merlin.targetgen.core_aten_profitability import prepare

    declaration, facts = inputs()
    (tmp_path / "model.mlir").write_text(str(source(6)))
    spec = tmp_path / "software.yaml"
    spec.write_text(yaml.safe_dump(dict(placement_cost_model=declaration)))
    report = dict(cases=[dict(status="bundled")])
    plan = prepare(
        tmp_path,
        report,
        lambda *a, **k: dict(status="host", refusal_code="independent_numeric_refusal"),
        software_spec=spec,
        facts=facts,
    )
    assert plan["placement_cost"]["strict_policy"] == dict(
        policy="strict_per_shape", status="not_applicable", reason_code="independent_numeric_refusal"
    )
    assert report["cases"][0]["numeric_admission"] is plan


def test_unknown_policy_is_not_silently_replaced(tmp_path, monkeypatch):
    import yaml

    from merlin.targetgen.core_aten_profitability import prepare

    declaration, facts = inputs()
    spec = tmp_path / "software.yaml"
    spec.write_text(yaml.safe_dump(dict(placement_cost_model=declaration)))
    monkeypatch.setenv("MERLIN_PLACEMENT_POLICY", "undefined")
    with pytest.raises(ValueError, match="unknown placement profitability policy"):
        prepare(
            tmp_path, dict(cases=[]), lambda *a, **k: pytest.fail("must not prepare"), software_spec=spec, facts=facts
        )


@pytest.mark.parametrize(
    "text,reason",
    [
        (
            """module { func.func @forward(%x: tensor<3x7xf32>, %e: tensor<7x3xf32>) -> tensor<7x3xf32> {
      %v=linalg.transpose ins(%x : tensor<3x7xf32>) outs(%e : tensor<7x3xf32>) permutation=[1,0]
      func.return %v : tensor<7x3xf32> }}""",
            "permuted_innermost_axis",
        ),
        (
            """module { func.func @forward(%x: tensor<3xf32>, %e: tensor<3x7xf32>) -> tensor<3x7xf32> {
      %v=linalg.broadcast ins(%x : tensor<3xf32>) outs(%e : tensor<3x7xf32>) dimensions=[1]
      func.return %v : tensor<3x7xf32> }}""",
            "broadcast_innermost_axis",
        ),
    ],
)
def test_named_inner_axis_selectors_are_not_contiguous_dma(text, reason):
    pattern, evidence, _ = movement_pattern(parse_mlir_text(text))
    assert pattern == "element_gather"
    assert evidence[0]["reason"] == reason


def test_static_index_arithmetic_is_not_device_computation():
    module = parse_mlir_text("""module {
      func.func @forward(%e: tensor<7xi64>) -> tensor<7xi64> {
        %r=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>], iterator_types=["parallel"]}
          outs(%e : tensor<7xi64>) {
          ^bb0(%unused: i64):
            %i=linalg.index 0 : index
            %v=arith.index_cast %i : index to i64
            linalg.yield %v : i64
          } -> tensor<7xi64>
        func.return %r : tensor<7xi64>
      }}""")
    declaration, facts = inputs()
    result = estimate_reference(module, candidate(module, "movement", "F1"), declaration, facts)
    assert result["reason_code"] == "index_generated_constant_host"
