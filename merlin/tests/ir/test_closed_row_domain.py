"""Independent row-domain admission/refusal and complete ordered tail plans."""

from dataclasses import replace

import pytest
from xdsl.dialects import func
from xdsl.dialects.builtin import UnitAttr

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.closed_row_domain import (
    RowBlock,
    analyze_closed_row_domain,
    plan_row_partition,
    validate_closed_row_domain,
)


def fixture(*, rows=5, reduction=7, columns=3, row_map="i", rhs_map="z,j", row_iterator="parallel", escape=False):
    a, b, c = f"tensor<{rows}x{reduction}xf32>", f"tensor<{reduction}x{columns}xf32>", f"tensor<{rows}x{columns}xf32>"
    extra = f"%escape = func.call @opaque(%a) : ({a}) -> {a}" if escape else ""
    module = parse_mlir_text(f"""module {{
      func.func private @opaque({a}) -> {a}
      func.func @test(%a: {a}, %b: {b}) -> {c} {{
        %zero = arith.constant 0.0 : f32
        %seed = tensor.splat %zero : {c}
        %result = linalg.generic {{indexing_maps = [affine_map<(i,j,z)->({row_map},z)>,
          affine_map<(i,j,z)->({rhs_map})>, affine_map<(i,j,z)->(i,j)>],
          iterator_types = ["{row_iterator}","parallel","reduction"]}}
          ins(%a,%b : {a},{b}) outs(%seed : {c}) {{
          ^bb0(%x: f32,%y: f32,%acc: f32):
            %p = arith.mulf %x,%y : f32
            %sum = arith.addf %acc,%p : f32
            linalg.yield %sum : f32
        }} -> {c}
        {extra}
        func.return %result : {c}
      }}
    }}""")
    f = next(o for o in module.body.block.ops if isinstance(o, func.FuncOp) and o.sym_name.data == "test")
    return module, f


def analyze(function, *, input_axes=(0, None), output_axes=(0,)):
    return analyze_closed_row_domain(
        inputs=function.body.block.args,
        outputs=function.body.block.last_op.operands,
        input_axes=input_axes,
        output_axes=output_axes,
    )


@pytest.mark.parametrize("dims", [(1, 1, 1), (5, 7, 3), (33, 2, 11)])
def test_retains_nonrow_reduction_and_scalar_source(dims):
    module, f = fixture(rows=dims[0], reduction=dims[1], columns=dims[2])
    before = str(module)
    d = analyze(f)
    assert d.rows == dims[0] and d.output_axes == (0,)
    assert d.operations[-1].reduction_dimensions == (2,)
    assert d.operations[-1].parallel_dimension == 0
    assert str(module) == before
    validate_closed_row_domain(d)


@pytest.mark.parametrize(
    "rows,block,sizes", [(1, 16, (1,)), (32, 16, (16, 16)), (33, 16, (16, 16, 1)), (17, 6, (6, 6, 5))]
)
def test_partition_covers_every_row_exactly_once(rows, block, sizes):
    p = plan_row_partition(rows, block)
    assert tuple(x.size for x in p.blocks) == sizes
    assert [i for x in p.blocks for i in range(x.offset, x.offset + x.size)] == list(range(rows))
    p.validate()
    with pytest.raises(ValueError, match="cover"):
        replace(p, blocks=(*p.blocks, RowBlock(0, 1))).validate()


@pytest.mark.parametrize("rows,block", [(0, 16), (-1, 2), (5, 0), (True, 1), (3, False), (3.0, 2)])
def test_invalid_static_domain_refuses(rows, block):
    with pytest.raises(ValueError, match="positive static"):
        plan_row_partition(rows, block)


@pytest.mark.parametrize("kind", ["row_reduce", "mixed_row", "invariant_indexed", "escape", "bad_axis"])
def test_nonlocal_or_unclosed_source_refuses(kind):
    options = {
        "row_reduce": dict(row_iterator="reduction"),
        "mixed_row": dict(row_map="i + 1"),
        "invariant_indexed": dict(rhs_map="i,j"),
        "escape": dict(escape=True),
        "bad_axis": {},
    }
    _, f = fixture(**options[kind])
    with pytest.raises(ValueError):
        analyze(f, input_axes=(1, None) if kind == "bad_axis" else (0, None))


def test_row_domain_cannot_discard_scalar_or_context_witness():
    _, f = fixture()
    d = analyze(f)
    with pytest.raises(ValueError, match="contents changed"):
        validate_closed_row_domain(replace(d, operations=()))
    next(o for o in f.walk() if o.name == "arith.mulf").attributes["source_changed"] = UnitAttr()
    with pytest.raises(ValueError, match="changed after analysis"):
        validate_closed_row_domain(d)


def test_original_row_index_observation_requires_offset_proof():
    module = parse_mlir_text("""module {
      func.func @indexed(%a: tensor<5xf32>) -> tensor<5xf32> {
        %zero = arith.constant 0.0 : f32
        %seed = tensor.splat %zero : tensor<5xf32>
        %result = linalg.generic {indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>],
          iterator_types = ["parallel"]} ins(%a: tensor<5xf32>) outs(%seed: tensor<5xf32>) {
          ^bb0(%x: f32, %unused: f32):
            %index = linalg.index 0 : index
            %integer = arith.index_cast %index : index to i32
            %value = arith.sitofp %integer : i32 to f32
            %sum = arith.addf %x, %value : f32
            linalg.yield %sum : f32
        } -> tensor<5xf32>
        func.return %result : tensor<5xf32>
      }
    }""")
    function = module.body.block.first_op
    with pytest.raises(ValueError, match="source"):
        analyze(function, input_axes=(0,))


def test_nested_scalar_region_requires_a_separate_source_proof():
    module = parse_mlir_text("""module {
      func.func @indexed(%a: tensor<5xf32>) -> tensor<5xf32> {
        %zero = arith.constant 0.0 : f32
        %seed = tensor.splat %zero : tensor<5xf32>
        %result = linalg.generic {indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>],
          iterator_types = ["parallel"]} ins(%a: tensor<5xf32>) outs(%seed: tensor<5xf32>) {
          ^bb0(%x: f32, %unused: f32):
            %condition = arith.constant true
            %value = scf.if %condition -> (f32) {
              scf.yield %x : f32
            } else {
              %fallback = arith.constant 0.0 : f32
              scf.yield %fallback : f32
            }
            %sum = arith.addf %x, %value : f32
            linalg.yield %sum : f32
        } -> tensor<5xf32>
        func.return %result : tensor<5xf32>
      }
    }""")
    module.verify()
    with pytest.raises(ValueError, match="unsupported source"):
        analyze(module.body.block.first_op, input_axes=(0,))


@pytest.mark.parametrize("observed", [False, True])
@pytest.mark.parametrize("cast", [False, True])
def test_empty_seed_is_admitted_only_when_its_contents_are_unobserved(observed, cast):
    body = "%sum = arith.addf %x, %seed_value : f32\nlinalg.yield %sum : f32" if observed else "linalg.yield %x : f32"
    seed = (
        "%uninitialized = tensor.empty() : tensor<5xf32>\n"
        "%seed = tensor.cast %uninitialized : tensor<5xf32> to tensor<5xf32>"
        if cast
        else "%seed = tensor.empty() : tensor<5xf32>"
    )
    module = parse_mlir_text(f"""module {{
      func.func @test(%a: tensor<5xf32>) -> tensor<5xf32> {{
        {seed}
        %result = linalg.generic {{indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>],
          iterator_types = ["parallel"]}} ins(%a: tensor<5xf32>) outs(%seed: tensor<5xf32>) {{
          ^bb0(%x: f32, %seed_value: f32):
            {body}
        }} -> tensor<5xf32>
        func.return %result : tensor<5xf32>
      }}
    }}""")
    module.verify()
    function = module.body.block.first_op
    if observed:
        with pytest.raises(ValueError):
            analyze(function, input_axes=(0,))
    else:
        validate_closed_row_domain(analyze(function, input_axes=(0,)))


def test_unused_output_does_not_hide_an_observed_duplicate_input():
    module = parse_mlir_text("""module {
      func.func @test(%a: tensor<5xf32>, %b: tensor<5xf32>) -> tensor<5xf32> {
        %result = linalg.generic {indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>, affine_map<(i)->(i)>],
          iterator_types = ["parallel"]} ins(%a,%b: tensor<5xf32>,tensor<5xf32>) outs(%b: tensor<5xf32>) {
          ^bb0(%x: f32, %rhs: f32, %unused: f32):
            %sum = arith.addf %x, %rhs : f32
            linalg.yield %sum : f32
        } -> tensor<5xf32>
        func.return %result : tensor<5xf32>
      }
    }""")
    module.verify()
    with pytest.raises(ValueError, match="invariant operand"):
        analyze(module.body.block.first_op)


@pytest.mark.parametrize("row_axis,reduced_axis", [(0, 1), (1, 0)])
def test_reduce_preserves_other_axes(row_axis, reduced_axis):
    source = "tensor<5x7xf32>" if row_axis == 0 else "tensor<7x5xf32>"
    module = parse_mlir_text(f"""module {{
      func.func @test(%a: {source}) -> tensor<5xf32> {{
        %zero = arith.constant 0.0 : f32
        %seed = tensor.splat %zero : tensor<5xf32>
        %r = "linalg.reduce"(%a,%seed) <{{dimensions = array<i64: {reduced_axis}>}}> ({{
          ^bb0(%x:f32,%y:f32):
            %z = arith.addf %x,%y : f32
            linalg.yield %z : f32
        }}) : ({source},tensor<5xf32>) -> tensor<5xf32>
        func.return %r : tensor<5xf32>
      }}
    }}""")
    f = next(o for o in module.body.block.ops if isinstance(o, func.FuncOp))
    d = analyze(f, input_axes=(row_axis,))
    assert d.operations[-1].reduction_dimensions == (reduced_axis,)
    validate_closed_row_domain(d)


@pytest.mark.parametrize("unit", [True, False])
def test_reshape_cannot_merge_rows_with_nonunit_axis(unit):
    shape = "tensor<1x5xf32>" if unit else "tensor<2x5xf32>"
    out = "tensor<5xf32>" if unit else "tensor<10xf32>"
    module = parse_mlir_text(f"""module {{func.func @test(%a: {shape}) -> {out} {{
      %r = tensor.collapse_shape %a [[0 : i64, 1 : i64]] : {shape} into {out}
      func.return %r : {out}
    }}}}""")
    f = next(o for o in module.body.block.ops if isinstance(o, func.FuncOp))
    if unit:
        validate_closed_row_domain(analyze(f, input_axes=(1,)))
    else:
        with pytest.raises(ValueError):
            analyze(f, input_axes=(1,))


def test_source_row_slice_must_preserve_full_extent():
    module = parse_mlir_text("""module {func.func @test(%a: tensor<5x7xf32>) -> tensor<4x7xf32> {
      %r = "tensor.extract_slice"(%a) <{static_offsets = array<i64: 1,0>, static_sizes = array<i64: 4,7>,
        static_strides = array<i64: 1,1>, operandSegmentSizes = array<i32: 1,0,0,0>}>
        : (tensor<5x7xf32>) -> tensor<4x7xf32>
      func.return %r : tensor<4x7xf32>
    }}""")
    f = next(o for o in module.body.block.ops if isinstance(o, func.FuncOp))
    with pytest.raises(ValueError):
        analyze(f, input_axes=(0,))
