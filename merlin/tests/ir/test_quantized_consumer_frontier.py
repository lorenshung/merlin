"""Typed source observations and ownership, independent of numeric admission."""

import pytest
from xdsl.dialects import func
from xdsl.dialects.builtin import UnitAttr

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.quantized_consumer_frontier import (
    analyze_quantized_consumer_frontier,
    validate_quantized_consumer_frontier,
)


def fixture(rows=2, channels=3, *, residual=False, unknown=False):
    source_type = f"tensor<{rows}x{channels}xbf16>"
    scale_type = f"tensor<{rows}xbf16>"
    integer_type = f"tensor<{rows}x{channels}xi8>"
    unknown_line = f"%other = func.call @opaque(%source) : ({source_type}) -> {source_type}" if unknown else ""
    source = "%other" if unknown else "%source"
    extra = ", %source" if residual else ""
    extra_type = ", " + source_type if residual else ""
    module = parse_mlir_text(f"""module {{
      func.func private @produce({source_type}) -> {source_type}
      func.func private @opaque({source_type}) -> {source_type}
      func.func @test(%input: {source_type}) -> ({integer_type}, {scale_type}{extra_type}) {{
        %source = func.call @produce(%input) : ({source_type}) -> {source_type}
        {unknown_line}
        %zero = arith.constant 0.0 : bf16
        %seed = tensor.splat %zero : {scale_type}
        %scale = "linalg.reduce"({source}, %seed) <{{dimensions = array<i64: 1>}}> ({{
          ^bb0(%a: bf16, %b: bf16):
            %maximum = arith.maximumf %a, %b : bf16
            linalg.yield %maximum : bf16
        }}) : ({source_type}, {scale_type}) -> {scale_type}
        %empty = tensor.empty() : {integer_type}
        %q = linalg.generic {{indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel", "parallel"]}} ins({source}, %scale : {source_type}, {scale_type}) outs(%empty : {integer_type}) {{
          ^bb0(%a: bf16, %s: bf16, %unused: i8):
            %normalized = arith.divf %a, %s : bf16
            %rounded = math.roundeven %normalized : bf16
            %i = arith.fptosi %rounded : bf16 to i8
            linalg.yield %i : i8
        }} -> {integer_type}
        func.return %q, %scale{extra} : {integer_type}, {scale_type}{extra_type}
      }}
    }}""")
    function = next(op for op in module.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "test")
    call = next(op for op in function.body.block.ops if isinstance(op, func.CallOp))
    returned = list(function.body.block.ops)[-1]
    return module, function, call.results[0], returned.operands[0]


@pytest.mark.parametrize("rows,channels", [(1, 7), (3, 5), (9, 2)])
def test_preserves_scale_and_integer_observations(rows, channels):
    module, function, source, integer = fixture(rows, channels)
    before = tuple(module.walk())
    frontier = analyze_quantized_consumer_frontier(integer, source_values=(source,))
    assert frontier.source_uses_closed
    assert frontier.observation_outputs == (integer, list(function.body.block.ops)[-1].operands[1])
    assert len(frontier.floating_escapes) == 1
    assert tuple(module.walk()) == before
    validate_quantized_consumer_frontier(frontier)


def test_retains_residual_source_escape():
    _, _, source, integer = fixture(residual=True)
    frontier = analyze_quantized_consumer_frontier(integer, source_values=(source,))
    assert not frontier.source_uses_closed
    assert source in frontier.observation_outputs
    assert frontier.unquantized_source_escapes == (source,)


def test_unknown_call_requires_numeric_and_effect_contract():
    _, _, source, integer = fixture(unknown=True)
    with pytest.raises(ValueError, match="unsupported numeric or effect"):
        analyze_quantized_consumer_frontier(integer, source_values=(source,))


def test_scalar_mutation_invalidates_retained_source():
    _, _, source, integer = fixture()
    frontier = analyze_quantized_consumer_frontier(integer, source_values=(source,))
    division = next(op for operation in frontier.operations for op in operation.walk() if op.name == "arith.divf")
    division.attributes["numeric_change"] = UnitAttr()
    with pytest.raises(ValueError, match="changed after analysis"):
        validate_quantized_consumer_frontier(frontier)


def test_enclosing_numeric_mutation_invalidates_retained_source():
    _, function, source, integer = fixture()
    frontier = analyze_quantized_consumer_frontier(integer, source_values=(source,))
    function.attributes["strictfp"] = UnitAttr()
    with pytest.raises(ValueError, match="changed after analysis"):
        validate_quantized_consumer_frontier(frontier)
    with pytest.raises(ValueError, match="enclosing strictfp"):
        analyze_quantized_consumer_frontier(integer, source_values=(source,))


def test_added_use_invalidates_retained_source():
    _, function, source, integer = fixture()
    frontier = analyze_quantized_consumer_frontier(integer, source_values=(source,))
    extra = func.CallOp("opaque", [source], [source.type])
    function.body.block.insert_op_before(extra, list(function.body.block.ops)[-1])
    with pytest.raises(ValueError, match="changed after analysis"):
        validate_quantized_consumer_frontier(frontier)


def assembly_fixture(*, overlap=False, gap=False):
    second = 1 if overlap else (3 if gap else 2)
    carrier_rows = 5 if gap else 4
    module = parse_mlir_text(f"""module {{
      func.func private @produce(tensor<2x3xbf16>) -> tensor<2x3xbf16>
      func.func @test(%x: tensor<2x3xbf16>) -> tensor<3x{carrier_rows}xi8> {{
        %a = func.call @produce(%x) : (tensor<2x3xbf16>) -> tensor<2x3xbf16>
        %b = func.call @produce(%x) : (tensor<2x3xbf16>) -> tensor<2x3xbf16>
        %empty = tensor.empty() : tensor<{carrier_rows}x3xbf16>
        %first = "tensor.insert_slice"(%a,%empty) <{{static_offsets = array<i64: 0,0>, static_sizes = array<i64: 2,3>, static_strides = array<i64: 1,1>, operandSegmentSizes = array<i32: 1,1,0,0,0>}}> : (tensor<2x3xbf16>, tensor<{carrier_rows}x3xbf16>) -> tensor<{carrier_rows}x3xbf16>
        %second = "tensor.insert_slice"(%b,%first) <{{static_offsets = array<i64: {second},0>, static_sizes = array<i64: 2,3>, static_strides = array<i64: 1,1>, operandSegmentSizes = array<i32: 1,1,0,0,0>}}> : (tensor<2x3xbf16>, tensor<{carrier_rows}x3xbf16>) -> tensor<{carrier_rows}x3xbf16>
        %tinit = tensor.empty() : tensor<3x{carrier_rows}xbf16>
        %transposed = linalg.transpose ins(%second : tensor<{carrier_rows}x3xbf16>) outs(%tinit : tensor<3x{carrier_rows}xbf16>) permutation = [1,0]
        %qinit = tensor.empty() : tensor<3x{carrier_rows}xi8>
        %q = linalg.generic {{indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]}} ins(%transposed : tensor<3x{carrier_rows}xbf16>) outs(%qinit : tensor<3x{carrier_rows}xi8>) {{
          ^bb0(%v: bf16,%unused: i8):
            %i = arith.fptosi %v : bf16 to i8
            linalg.yield %i : i8
        }} -> tensor<3x{carrier_rows}xi8>
        func.return %q : tensor<3x{carrier_rows}xi8>
      }}
    }}""")
    f = next(op for op in module.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "test")
    calls = [op for op in f.body.block.ops if isinstance(op, func.CallOp)]
    integer = list(f.body.block.ops)[-1].operands[0]
    return module, f, tuple(call.results[0] for call in calls), integer


def test_complete_assembly_and_transpose_coordinates():
    _, _, sources, integer = assembly_fixture()
    frontier = analyze_quantized_consumer_frontier(integer, source_values=sources)
    assert frontier.source_uses_closed
    assert [(offsets, sizes) for _, offsets, sizes in frontier.assembly_boxes] == [((0, 0), (2, 3)), ((2, 0), (2, 3))]
    validate_quantized_consumer_frontier(frontier)


@pytest.mark.parametrize("option,message", [("overlap", "overlap"), ("gap", "every carrier")])
def test_unowned_assembly_refuses(option, message):
    _, _, sources, integer = assembly_fixture(**{option: True})
    with pytest.raises(ValueError, match=message):
        analyze_quantized_consumer_frontier(integer, source_values=sources)


def test_zero_source_domain_refuses():
    _, _, source, integer = fixture(0, 3)
    with pytest.raises(ValueError, match="static floating producer"):
        analyze_quantized_consumer_frontier(integer, source_values=(source,))


def test_assembly_coordinate_mutation_invalidates_witness():
    from xdsl.dialects.builtin import DenseArrayBase, i64

    _, _, sources, integer = assembly_fixture()
    frontier = analyze_quantized_consumer_frontier(integer, source_values=sources)
    insertion = next(op for op in frontier.operations if op.name == "tensor.insert_slice")
    insertion.properties["static_offsets"] = DenseArrayBase.from_list(i64, [1, 0])
    with pytest.raises(ValueError, match="changed after analysis"):
        validate_quantized_consumer_frontier(frontier)


def test_new_scale_use_invalidates_witness():
    _, function, source, integer = fixture()
    frontier = analyze_quantized_consumer_frontier(integer, source_values=(source,))
    scale = frontier.floating_escapes[0]
    extra = func.CallOp("observe_scale", [scale], [])
    function.body.block.insert_op_before(extra, list(function.body.block.ops)[-1])
    with pytest.raises(ValueError, match="changed after analysis"):
        validate_quantized_consumer_frontier(frontier)


def test_semantic_fingerprint_retains_coordinates_and_dimensions():
    from merlin.llvmlower.quantized_consumer_frontier import quantized_consumer_semantic_sha256

    _, _, source, integer = fixture(2, 3)
    first = quantized_consumer_semantic_sha256(analyze_quantized_consumer_frontier(integer, source_values=(source,)))
    _, _, source, integer = fixture(2, 3)
    same = quantized_consumer_semantic_sha256(analyze_quantized_consumer_frontier(integer, source_values=(source,)))
    _, _, source, integer = fixture(2, 5)
    different = quantized_consumer_semantic_sha256(
        analyze_quantized_consumer_frontier(integer, source_values=(source,))
    )
    assert first == same and first != different


def test_semantic_fingerprint_ignores_only_provenance():
    from xdsl.dialects.builtin import StringAttr

    from merlin.llvmlower.quantized_consumer_frontier import quantized_consumer_semantic_sha256

    _, _, source, integer = fixture()
    original = quantized_consumer_semantic_sha256(analyze_quantized_consumer_frontier(integer, source_values=(source,)))
    integer.owner.attributes["prov.region_id"] = StringAttr("different_source_instance")
    renamed = quantized_consumer_semantic_sha256(analyze_quantized_consumer_frontier(integer, source_values=(source,)))
    assert original == renamed
    integer.owner.attributes["numeric_policy"] = StringAttr("changed")
    changed = quantized_consumer_semantic_sha256(analyze_quantized_consumer_frontier(integer, source_values=(source,)))
    assert changed != original


def test_discovery_uses_actual_dependency_without_source_label():
    from merlin.llvmlower.quantized_consumer_frontier import find_quantized_consumer_frontiers

    module, _, source, integer = fixture(3, 7)
    frontiers = find_quantized_consumer_frontiers(module, source_values=(source,))
    assert len(frontiers) == 1 and frontiers[0].integer_output is integer
    assert frontiers[0].source_values == (source,)


def test_discovery_keeps_residual_refusal():
    from merlin.llvmlower.quantized_consumer_frontier import find_quantized_consumer_frontiers

    module, _, source, _ = fixture(residual=True)
    frontiers = find_quantized_consumer_frontiers(module, source_values=(source,))
    assert len(frontiers) == 1 and not frontiers[0].source_uses_closed


def test_discovery_does_not_cross_unknown_call():
    from merlin.llvmlower.quantized_consumer_frontier import find_quantized_consumer_frontiers

    module, _, source, _ = fixture(unknown=True)
    assert find_quantized_consumer_frontiers(module, source_values=(source,)) == ()


def test_discovery_reordered_sources_preserve_actual_assembly():
    from merlin.llvmlower.quantized_consumer_frontier import find_quantized_consumer_frontiers

    module, _, sources, integer = assembly_fixture()
    (frontier,) = find_quantized_consumer_frontiers(module, source_values=tuple(reversed(sources)))
    assert frontier.integer_output is integer
    assert frontier.source_values == tuple(reversed(sources))
    assert [(value, offsets) for value, offsets, _ in frontier.assembly_boxes] == [
        (sources[0], (0, 0)),
        (sources[1], (2, 0)),
    ]
