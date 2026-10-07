"""Source consumers must close before any observational writer replacement."""

from dataclasses import replace

import pytest
from xdsl.dialects import builtin, func
from xdsl.ir import Block

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.closed_group_writer import ClosedGroupWriterContract, source_function_semantic_sha256
from merlin.llvmlower.consumer_observed_group_writer import (
    ConsumerObservationWitness,
    install_consumer_observed_group_writers,
)
from merlin.llvmlower.ordered_bf16_group_binding import SourceExactGroupPreparation
from merlin.llvmlower.quantized_consumer_frontier import (
    analyze_quantized_consumer_frontier,
    quantized_consumer_semantic_sha256,
)
from merlin.tests.ir.test_ordered_fma_groups import contraction, module, narrow, scale
from merlin.xdsl_dialects._common import text


def prepare(tmp_path, *, residual=False, different_source=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    block = Block(arg_types=[builtin.TensorType(builtin.bf16, [2, 3, 4]), builtin.TensorType(builtin.bf16, [2, 4, 5])])
    floating = []
    integer = []
    for index in range(2):
        contraction_value = contraction(block, *block.args).results[0]
        if different_source and index == 1:
            contraction_value = scale(block, contraction_value).results[0]
        result = narrow(block, contraction_value).results[0]
        floating.append(result)
        template = parse_mlir_text("""module {func.func @cast(%x:tensor<2x3x5xbf16>)->tensor<2x3x5xi8> {
          %init=tensor.empty() : tensor<2x3x5xi8>
          %q=linalg.generic {indexing_maps=[affine_map<(d0,d1,d2)->(d0,d1,d2)>,affine_map<(d0,d1,d2)->(d0,d1,d2)>],iterator_types=["parallel","parallel","parallel"]} ins(%x:tensor<2x3x5xbf16>) outs(%init:tensor<2x3x5xi8>) {
            ^bb0(%v:bf16,%unused:i8):
              %i=arith.fptosi %v : bf16 to i8
              linalg.yield %i : i8
          } -> tensor<2x3x5xi8>
          func.return %q : tensor<2x3x5xi8>
        }}""")
        body = list(template.body.block.ops)[0].body.block
        mapping = {body.args[0]: result}
        for op in list(body.ops)[:-1]:
            clone = op.clone(value_mapper=mapping)
            block.add_op(clone)
            mapping.update(zip(op.results, clone.results))
        integer.append(mapping[list(body.ops)[-1].operands[0]])
    outputs = [*integer, *([floating[0]] if residual else [])]
    original = tmp_path / "source.mlir"
    original.write_text(text(module(block, outputs)))
    preparation = SourceExactGroupPreparation()
    selected = preparation(original, tmp_path / "prepared")
    ir = parse_mlir_text(selected.read_text())
    f = next(op for op in ir.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "forward")
    calls = [op for op in f.body.block.ops if isinstance(op, func.CallOp)]
    returned = list(f.body.block.ops)[-1]
    frontiers = [
        analyze_quantized_consumer_frontier(returned.operands[i], source_values=(calls[i].results[0],))
        for i in range(2)
    ]
    functions = {op.sym_name.data: op for op in ir.body.block.ops if isinstance(op, func.FuncOp)}
    contracts = {
        r["symbol"]: ClosedGroupWriterContract(
            source_function_semantic_sha256(functions[r["symbol"]]),
            "external_group",
            "1" * 64,
            "2" * 64,
            "exact_consumer_observations",
            "3" * 64,
            64,
            True,
            True,
            True,
            True,
            "rne_returned_values",
            "native_functional_screen",
        )
        for r in preparation.receipt["records"]
    }
    witnesses = [
        ConsumerObservationWitness(
            frontier,
            quantized_consumer_semantic_sha256(frontier),
            tuple(str(v.type) for v in frontier.observation_outputs),
            "2" * 64,
            "4" * 64,
            "rne_returned_values",
        )
        for frontier in frontiers
    ]
    return ir, preparation.receipt, contracts, witnesses


def test_complete_live_consumers_allow_one_shared_writer(tmp_path):
    ir, receipt, contracts, witnesses = prepare(tmp_path)
    result = install_consumer_observed_group_writers(ir, receipt, contracts, consumer_witnesses=witnesses)
    assert result["consumer_observation_calls"] == result["source_groups"] == 2
    assert result["physical_writers"] == 1
    assert sum(op.name == "arith.fptosi" for op in ir.walk()) == 2
    ir.verify()


@pytest.mark.parametrize(
    "field,value",
    [
        ("consumer_semantic_sha256", "0" * 64),
        ("observed_types", ()),
        ("numerical_witness_sha256", "5" * 64),
        ("effect_witness_sha256", "bad"),
        ("fenv_policy", "unknown"),
    ],
)
def test_missing_consumer_obligation_refuses_before_mutation(tmp_path, field, value):
    ir, receipt, contracts, witnesses = prepare(tmp_path)
    before = text(ir)
    witnesses[0] = replace(witnesses[0], **{field: value})
    with pytest.raises(ValueError):
        install_consumer_observed_group_writers(ir, receipt, contracts, consumer_witnesses=witnesses)
    assert text(ir) == before


def test_residual_bf16_escape_cannot_be_dropped(tmp_path):
    ir, receipt, contracts, witnesses = prepare(tmp_path, residual=True)
    before = text(ir)
    with pytest.raises(ValueError, match="unquantized escape"):
        install_consumer_observed_group_writers(ir, receipt, contracts, consumer_witnesses=witnesses)
    assert text(ir) == before


def test_missing_selected_call_coverage_refuses(tmp_path):
    ir, receipt, contracts, witnesses = prepare(tmp_path)
    before = text(ir)
    with pytest.raises(ValueError, match="lacks complete consumer"):
        install_consumer_observed_group_writers(ir, receipt, contracts, consumer_witnesses=witnesses[:1])
    assert text(ir) == before


def test_changed_context_refuses_before_writer_install(tmp_path):
    ir, receipt, contracts, witnesses = prepare(tmp_path)
    witnesses[0].frontier.integer_output.owner.attributes["strictfp"] = builtin.UnitAttr()
    before = text(ir)
    with pytest.raises(ValueError, match="changed after analysis"):
        install_consumer_observed_group_writers(ir, receipt, contracts, consumer_witnesses=witnesses)
    assert text(ir) == before


def test_equal_shape_foreign_frontier_is_not_source_binding(tmp_path):
    ir, receipt, contracts, _ = prepare(tmp_path / "a")
    _, _, _, foreign = prepare(tmp_path / "b")
    before = text(ir)
    with pytest.raises(ValueError, match="foreign source"):
        install_consumer_observed_group_writers(ir, receipt, contracts, consumer_witnesses=foreign)
    assert text(ir) == before


def preparation_contracts(tmp_path, *, residual=False, different_source=False):
    from merlin.llvmlower.consumer_observed_group_writer import (
        ConsumerObservationContract,
        ConsumerObservedGroupPreparation,
    )

    ir, receipt, contracts, witnesses = prepare(tmp_path, residual=residual, different_source=different_source)
    provider = next(iter(contracts.values()))
    w = witnesses[-1] if residual else witnesses[0]
    proof = ConsumerObservationContract(
        w.consumer_semantic_sha256, w.observed_types, w.numerical_witness_sha256, w.effect_witness_sha256, w.fenv_policy
    )
    return ir, receipt, provider, proof, ConsumerObservedGroupPreparation


def test_normal_callback_matches_complete_source_and_consumer(tmp_path):
    _, _, provider, proof, Preparation = preparation_contracts(tmp_path)
    transform = Preparation([provider], [proof])
    output = transform(tmp_path / "source.mlir", tmp_path / "normal")
    actual = parse_mlir_text(output.read_text())
    actual.verify()
    assert transform.receipt["consumer_observation_calls"] == 2
    assert transform.receipt["source_cpu_groups"] == 0
    assert transform.receipt["physical_writers"] == 1
    assert sum(op.name == "arith.fptosi" for op in actual.walk()) == 2


def test_preparation_different_consumer_proof_keeps_actual_source(tmp_path):
    ir, receipt, provider, proof, Preparation = preparation_contracts(tmp_path)
    before = text(ir)
    transform = Preparation([provider], [replace(proof, consumer_semantic_sha256="0" * 64)])
    result = transform.install(ir, receipt)
    assert result["consumer_observation_calls"] == 0 and result["source_cpu_groups"] == 2
    assert text(ir) == before


def test_preparation_residual_use_keeps_actual_source(tmp_path):
    ir, receipt, provider, proof, Preparation = preparation_contracts(tmp_path, residual=True)
    transform = Preparation([provider], [proof])
    result = transform.install(ir, receipt)
    assert result["consumer_observation_calls"] == 1 and result["source_cpu_groups"] == 1
    assert sum(op.name == "arith.fptosi" for op in ir.walk()) == 2
    assert any(op.name == "math.fma" for op in ir.walk())


@pytest.mark.parametrize("kind", ["producer", "consumer"])
def test_preparation_ambiguous_proofs_refuse_before_mutation(tmp_path, kind):
    ir, _, provider, proof, Preparation = preparation_contracts(tmp_path)
    before = text(ir)
    with pytest.raises(ValueError, match="ambiguous complete"):
        Preparation(
            [provider, provider] if kind == "producer" else [provider],
            [proof, proof] if kind == "consumer" else [proof],
        )
    assert text(ir) == before


def test_preparation_mutated_ancestor_refuses_before_mutation(tmp_path):
    ir, receipt, provider, proof, Preparation = preparation_contracts(tmp_path)
    transform = Preparation([provider], [proof])
    ir.attributes["strictfp"] = builtin.UnitAttr()
    before = text(ir)
    with pytest.raises(ValueError, match="context changed"):
        transform.install(ir, receipt)
    assert text(ir) == before


def test_preparation_equal_shapes_different_source_semantics_stay_separate(tmp_path):
    ir, receipt, provider, proof, Preparation = preparation_contracts(tmp_path, different_source=True)
    result = Preparation([provider], [proof]).install(ir, receipt)
    assert result["consumer_observation_calls"] == 1 and result["source_cpu_groups"] == 1
    assert result["compatible_source_producers"] == 1
    assert sum(op.name == "arith.mulf" for op in ir.walk()) == 1
    assert sum(op.name == "math.fma" for op in ir.walk()) == 1


def test_normal_callback_cached_source_owns_its_output_directory(tmp_path):
    _, receipt, provider, proof, Preparation = preparation_contracts(tmp_path)

    class CachedPreparation:
        def __init__(self):
            self.receipt = receipt

        def __call__(self, source_path, workdir):
            return receipt["selected_path"]

    destination = tmp_path / "new" / "normal"
    transform = Preparation([provider], [proof], source_preparation=CachedPreparation())
    output = transform(tmp_path / "source.mlir", destination)
    assert output.is_file() and (destination / "consumer_observed_groups.json").is_file()
    assert transform.receipt["consumer_observation_calls"] == 2


def test_normal_observed_callback_reuses_one_explicit_private_workspace(tmp_path):
    from xdsl.dialects import memref

    from merlin.tests.ir.test_private_writer_workspace import workspace

    ir, receipt, provider, proof, Preparation = preparation_contracts(tmp_path)
    original = next(
        op for op in ir.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "forward"
    ).function_type
    transform = Preparation(
        [replace(provider, private_workspaces=(workspace(),))], [proof], reuse_private_workspaces=True
    )
    result = transform.install(ir, receipt)
    assert result["consumer_observation_calls"] == 2
    pooled = result["private_workspace_pool"]
    assert pooled["allocations"] == 1 and pooled["required_bytes"] == [37]
    assert pooled["root_calls"] == 2
    owner = next(op for op in ir.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "forward")
    assert owner.function_type == original
    assert sum(isinstance(op, memref.AllocOp) for op in owner.body.block.ops) == 1
    ir.verify()


@pytest.mark.parametrize("invalid_path", ["multiple_owners", "symbolic_escape"])
def test_combined_workspace_preflight_refuses_before_any_producer_replacement(tmp_path, invalid_path):
    from xdsl.dialects import arith

    from merlin.tests.ir.test_private_writer_workspace import workspace

    ir, receipt, provider, proof, Preparation = preparation_contracts(tmp_path)
    if invalid_path == "multiple_owners":
        original = parse_mlir_text((tmp_path / "source.mlir").read_text())
        other = original.body.block.first_op.clone()
        other.sym_name = builtin.StringAttr("second_public_owner")
        original.body.block.add_op(other)
        path = tmp_path / "two_owners.mlir"
        path.write_text(text(original))
        source = SourceExactGroupPreparation()
        prepared = source(path, tmp_path / "two_owners_prepared")
        ir = parse_mlir_text(prepared.read_text())
        receipt = source.receipt
    else:
        owner = next(op for op in ir.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "forward")
        marker = arith.ConstantOp.from_int_and_width(0, 32)
        marker.attributes["address"] = builtin.ArrayAttr(
            [builtin.DictionaryAttr({"callback": builtin.SymbolRefAttr(receipt["records"][0]["symbol"])})]
        )
        owner.body.block.insert_op_before(marker, owner.body.block.first_op)
    transform = Preparation(
        [replace(provider, private_workspaces=(workspace(),))], [proof], reuse_private_workspaces=True
    )
    before = text(ir)
    with pytest.raises(ValueError, match="common public owner|symbolic escape"):
        transform.install(ir, receipt)
    assert text(ir) == before
    assert not any(op.name == "memref.alloc" for op in ir.walk())
    assert sum(op.name == "math.fma" for op in ir.walk()) == len(receipt["records"])


def test_pooled_maxima_overflow_refuses_before_any_producer_replacement(tmp_path):
    from merlin.llvmlower.consumer_observed_group_writer import (
        ConsumerObservationContract,
        ConsumerObservedGroupPreparation,
    )
    from merlin.tests.ir.test_private_writer_workspace import workspace

    ir, receipt, contracts, witnesses = prepare(tmp_path, different_source=True)
    large, small = workspace(1 << 62, 1), workspace(1, 1)
    providers = [
        replace(
            value,
            provider_symbol="provider_" + str(index),
            private_workspaces=(large, small) if index == 0 else (small, large),
        )
        for index, value in enumerate(contracts.values())
    ]
    witness = witnesses[0]
    proof = ConsumerObservationContract(
        witness.consumer_semantic_sha256,
        witness.observed_types,
        witness.numerical_witness_sha256,
        witness.effect_witness_sha256,
        witness.fenv_policy,
    )
    preparation = ConsumerObservedGroupPreparation(providers, [proof], reuse_private_workspaces=True)
    before = text(ir)
    with pytest.raises(ValueError, match="pooled private workspace allocation overflows"):
        preparation.install(ir, receipt)
    assert text(ir) == before
    assert sum(op.name == "math.fma" for op in ir.walk()) == 2
    assert not any(op.name == "memref.alloc" for op in ir.walk())
