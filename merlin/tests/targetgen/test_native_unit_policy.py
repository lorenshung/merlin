"""Native roles are derived from source scalar dataflow, never overload names."""

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen.native_unit_policy import construction_evidence


def source(expression, *, literal=False, reduction=False):
    return parse_mlir_text(
        '''module {
      func.func @forward(%x: tensor<3xf32>,%y: tensor<3xf32>) -> tensor<3xf32> {
        %zero=arith.constant 0.0 : f32
        %one=arith.constant 1.0 : f32
        %ones=tensor.splat %one : tensor<3xf32>
        %e=tensor.empty() : tensor<3xf32>
        %r=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],
          iterator_types=["'''
        + ("reduction" if reduction else "parallel")
        + """"]}
          ins(%x,"""
        + ("%ones" if literal else "%y")
        + """ : tensor<3xf32>,tensor<3xf32>) outs(%e : tensor<3xf32>) {
          ^bb0(%a: f32,%b: f32,%c: f32):
            """
        + expression
        + """
            linalg.yield %v : f32
          } -> tensor<3xf32>
        func.return %r : tensor<3xf32>
      }}"""
    )


def test_parallel_runtime_product_is_not_a_native_contraction():
    result = construction_evidence(source("%v=arith.mulf %a,%b : f32"))
    assert result["status"] == "host"
    assert result["evidence"][0]["reason"] == "parallel_runtime_product_construction"


def test_literal_scale_and_own_linear_math_are_native():
    assert construction_evidence(source("%v=arith.mulf %a,%b : f32", literal=True))["status"] == "eligible"
    assert construction_evidence(source("%v=arith.addf %a,%b : f32"))["status"] == "eligible"


def test_true_reduction_product_retains_contraction_role():
    assert construction_evidence(source("%v=arith.mulf %a,%b : f32", reduction=True))["status"] == "eligible"


def test_rounding_cannot_be_relabelled_as_component_math():
    result = construction_evidence(source("%v=math.floor %a : f32"))
    assert result["status"] == "host"
    assert result["evidence"][0]["reason"] == "nonlinear_rounding_construction"


def test_literal_expression_is_not_mistaken_for_runtime_multiplier():
    assert (
        construction_evidence(source("%n=arith.negf %b : f32\n%v=arith.mulf %a,%n : f32", literal=True))["status"]
        == "eligible"
    )


def test_index_arithmetic_is_not_accelerated_by_copying_its_host_literal_table():
    ramp = source("%i=linalg.index 0 : index\n%j=arith.index_cast %i : index to i64\n%v=arith.sitofp %j : i64 to f32")
    literal = source("%i=linalg.index 0 : index\n%v=arith.constant 2.0 : f32")
    assert construction_evidence(ramp)["evidence"][0]["reason"] == "index_generated_payload_arithmetic"
    assert construction_evidence(literal)["status"] == "eligible"


def test_structured_address_selection_is_movement_not_index_payload_math():
    module = parse_mlir_text("""module {
      func.func @forward(%x: tensor<3xf32>,%y: tensor<3xf32>) -> tensor<3xf32> {
        %e=tensor.empty() : tensor<3xf32>
        %r=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
          outs(%e : tensor<3xf32>) {
          ^bb0(%unused: f32):
            %i=linalg.index 0 : index
            %z=arith.constant 0 : index
            %p=arith.cmpi eq,%i,%z : index
            %v=scf.if %p -> (f32) {
              %a=tensor.extract %x[%i] : tensor<3xf32>
              scf.yield %a : f32
            } else {
              %b=tensor.extract %y[%i] : tensor<3xf32>
              scf.yield %b : f32
            }
            linalg.yield %v : f32
          } -> tensor<3xf32>
        func.return %r : tensor<3xf32>
      }}""")
    assert construction_evidence(module)["status"] == "eligible"


def test_shared_index_dag_is_analyzed_without_expanding_its_tree():
    lines = ["%i=linalg.index 0 : index"]
    previous = "%i"
    for index in range(70):
        current = f"%d{index}"
        lines.append(f"{current}=arith.addi {previous},{previous} : index")
        previous = current
    lines.extend([f"%n=arith.index_cast {previous} : index to i64", "%v=arith.sitofp %n : i64 to f32"])
    result = construction_evidence(source("\n".join(lines)))
    assert result["status"] == "host"


def test_exclusions_are_selected_by_the_targets_numeric_contract():
    module = source("%v=arith.mulf %a,%b : f32")
    assert construction_evidence(module, excluded=[])["status"] == "eligible"
    decision = construction_evidence(module, excluded=["parallel_runtime_product_construction"])
    assert decision["status"] == "host"
    assert decision["classified_constructions"] == decision["evidence"]


def test_precomputed_result_tables_do_not_accelerate_their_construction():
    def table(values):
        return parse_mlir_text(
            """module {
          func.func @forward() -> tensor<3xf32> {
            %v=arith.constant dense<"""
            + values
            + """> : tensor<3xf32>
            func.return %v : tensor<3xf32>
          }}"""
        )

    generated = construction_evidence(table("[1.0,3.0,5.0]"))
    assert generated["status"] == "host"
    assert generated["evidence"][0]["reason"] == "nonuniform_literal_result_construction"
    assert construction_evidence(table("2.0"))["status"] == "eligible"
    assert construction_evidence(table("[1.0,3.0,5.0]"), excluded=[])["status"] == "eligible"


def test_literal_address_tables_are_native_movement():
    module = parse_mlir_text("""module {
      func.func @forward(%x: tensor<3xf32>) -> tensor<3xf32> {
        %table=arith.constant dense<[2,1,0]> : tensor<3xindex>
        %e=tensor.empty() : tensor<3xf32>
        %r=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
          outs(%e : tensor<3xf32>) {
          ^bb0(%unused: f32):
            %i=linalg.index 0 : index
            %j=tensor.extract %table[%i] : tensor<3xindex>
            %a=tensor.extract %x[%j] : tensor<3xf32>
            linalg.yield %a : f32
          } -> tensor<3xf32>
        func.return %r : tensor<3xf32>
      }}""")
    assert construction_evidence(module)["status"] == "eligible"


def test_literal_uniformity_uses_storage_bits_for_nan_and_zero():
    from xdsl.dialects import arith, func
    from xdsl.dialects.builtin import BytesAttr, DenseIntOrFPElementsAttr, ModuleOp, TensorType, f32

    def table(words):
        data = DenseIntOrFPElementsAttr(
            TensorType(f32, [len(words)]), BytesAttr(b"".join(w.to_bytes(4, "little") for w in words))
        )
        constant = arith.ConstantOp(data)
        function = func.FuncOp("forward", ([], [constant.result.type]))
        function.body.block.add_ops([constant, func.ReturnOp(constant.result)])
        return ModuleOp([function])

    assert construction_evidence(table([0x7FC01234, 0x7FC01234]))["status"] == "eligible"
    assert construction_evidence(table([0, 0x80000000]))["status"] == "host"
    assert construction_evidence(table([0x7FC01234, 0x7FC05678]))["status"] == "host"
    assert construction_evidence(table([]))["status"] == "eligible"


def test_saturation_used_as_comparator_is_excluded_but_plain_clamp_is_native():
    clamp = "%z=arith.constant 0.0 : f32\n%capped=arith.maximumf %a,%z : f32\n"
    predicate = "%p=arith.cmpf ogt,%capped,%z : f32\n%v=arith.uitofp %p : i1 to f32"
    result = construction_evidence(source(clamp + predicate))
    assert "saturation_comparators" in {e["reason"] for e in result["evidence"]}
    assert construction_evidence(source(clamp + "%v=arith.addf %capped,%z : f32"))["status"] == "eligible"
    assert (
        construction_evidence(source("%p=arith.cmpf ogt,%a,%b : f32\n%v=arith.select %p,%a,%b : f32"))["status"]
        == "eligible"
    )


def test_biased_rounding_is_detected_without_excluding_an_ordinary_affine_offset():
    bias = "%half=arith.constant 0.5 : f32\n%biased=arith.addf %a,%half : f32\n"
    result = construction_evidence(source(bias + "%v=math.floor %biased : f32"))
    assert "readout_offset_rounding" in {e["reason"] for e in result["evidence"]}
    narrowing = "%n=arith.fptosi %biased : f32 to i32\n%v=arith.sitofp %n : i32 to f32"
    assert construction_evidence(source(bias + narrowing), excluded=["readout_offset_rounding"])["status"] == "host"
    assert (
        construction_evidence(source("%half=arith.constant 0.5 : f32\n%v=arith.addf %a,%half : f32"))["status"]
        == "eligible"
    )
    assert (
        construction_evidence(source("%n=arith.fptosi %a : f32 to i32\n%v=arith.sitofp %n : i32 to f32"))["status"]
        == "eligible"
    )


def test_magic_bias_rounding_is_detected():
    module = source(
        "%bias=arith.constant 16777216.0 : f32\n%shift=arith.addf %a,%bias : f32\n%v=arith.subf %shift,%bias : f32"
    )
    assert construction_evidence(module, excluded=["readout_offset_rounding"])["status"] == "host"


def test_rectifier_extremum_identity_is_excluded_but_native_relu_is_not():
    zero = "%z=arith.constant 0.0 : f32\n"
    identity = "%d=arith.subf %a,%b : f32\n%r=arith.maximumf %d,%z : f32\n%v=arith.addf %r,%b : f32"
    assert construction_evidence(source(zero + identity), excluded=["multi_op_identity"])["status"] == "host"
    assert construction_evidence(source(zero + "%v=arith.maximumf %a,%z : f32"))["status"] == "eligible"
    assert construction_evidence(source("%v=arith.maximumf %a,%b : f32"))["status"] == "eligible"


def boolean_source():
    return parse_mlir_text("""module {
      func.func @forward(%x: tensor<3xi1>,%y: tensor<3xi1>) -> tensor<3xi1> {
        %e=tensor.empty() : tensor<3xi1>
        %r=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,
          affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
          ins(%x,%y : tensor<3xi1>,tensor<3xi1>) outs(%e : tensor<3xi1>) {
          ^bb0(%a: i1,%b: i1,%c: i1):
            %v=arith.ori %a,%b : i1
            linalg.yield %v : i1
          } -> tensor<3xi1>
        func.return %r : tensor<3xi1>
      }}""")


def test_boolean_or_is_host_and_is_never_rewritten_as_add_and_truth():
    from merlin.llvmlower.boolean_components import rewrite_boolean_components

    module = boolean_source()
    original = str(module)
    assert construction_evidence(module, excluded=["multi_op_identity"])["status"] == "host"
    assert rewrite_boolean_components(module) == ({}, 0)
    assert str(module) == original


def test_boolean_add_truth_identity_is_detected_after_rewriting():
    module = boolean_source()
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerAttr, i32
    from xdsl.rewriter import Rewriter

    union = next(op for op in module.walk() if op.name == "arith.ori")
    left, right = (arith.ExtUIOp(v, i32) for v in union.operands)
    add = arith.AddiOp(left.result, right.result)
    zero = arith.ConstantOp(IntegerAttr(0, i32))
    truth = arith.CmpiOp(add.result, zero.result, "ne")
    Rewriter.replace_op(union, [left, right, add, zero, truth], [truth.result])
    assert construction_evidence(module, excluded=["multi_op_identity"])["status"] == "host"


def test_an_unrelated_reduction_does_not_license_parallel_products_or_index_payloads():
    from xdsl.dialects.builtin import ModuleOp

    reduction = source("%v=arith.mulf %a,%b : f32", reduction=True)
    for other, reason in [
        (source("%v=arith.mulf %a,%b : f32"), "parallel_runtime_product_construction"),
        (
            source("%i=linalg.index 0 : index\n%j=arith.index_cast %i : index to i64\n%v=arith.sitofp %j : i64 to f32"),
            "index_generated_payload_arithmetic",
        ),
    ]:
        module = ModuleOp([op.clone() for m in [reduction, other] for op in m.body.block.ops])
        assert reason in {e["reason"] for e in construction_evidence(module)["evidence"]}


def test_deep_shared_payloads_do_not_expand_exponentially_or_hit_python_recursion():
    lines = ["%c0=arith.constant 1.0 : f32"]
    for i in range(1, 1200):
        lines.append(f"%c{i}=arith.addf %c{i - 1},%c{i - 1} : f32")
    lines.append("%v=arith.mulf %a,%c1199 : f32")
    assert construction_evidence(source("\n".join(lines)))["status"] == "eligible"


def test_integer_offset_readout_is_rounding():
    module = parse_mlir_text("""module {
      func.func @forward(%a: i64) -> i64 {
        %offset=arith.constant 128 : i64
        %bits=arith.constant 8 : i64
        %biased=arith.addi %a,%offset : i64
        %shifted=arith.shrsi %biased,%bits : i64
        func.return %shifted : i64
      }}""")
    assert construction_evidence(module, excluded=["readout_offset_rounding"])["status"] == "host"


def test_readout_uses_only_the_current_module_during_prelude_rewriting():
    module = boolean_source()
    generic = next(op for op in module.walk() if op.name == "linalg.generic")
    # A replaced generic can retain stale SSA uses after its region is removed.
    # It is absent from the current serialized program and has no readout role.
    detached = generic.clone(value_mapper={generic.inputs[0]: generic.results[0]})
    detached.detach_region(0)
    assert construction_evidence(module, excluded=["multi_op_identity"])["status"] == "host"


@pytest.mark.parametrize(
    "carrier,result_type",
    [
        (
            (
                "%r=linalg.generic {indexing_maps=[affine_map<(d0,d1)->(d0,d1)>],iterat"
                'or_types=["parallel","parallel"]} outs(%table:tensor<2x2xi32>) { ^bb0('
                "%old:i32): linalg.yield %old:i32 } -> tensor<2x2xi32>"
            ),
            "2x2xi32",
        ),
        (
            (
                "%r=linalg.generic {indexing_maps=[affine_map<(d0,d1)->(d0,d1)>,affine_"
                'map<(d0,d1)->(d0,d1)>],iterator_types=["parallel","parallel"]} ins(%x:'
                "tensor<2x2xi32>) outs(%table:tensor<2x2xi32>) { ^bb0(%unused:i32,%old:"
                "i32): linalg.yield %old:i32 } -> tensor<2x2xi32>"
            ),
            "2x2xi32",
        ),
        (
            (
                "%i=arith.constant 0:index\n%a=tensor.extract %table[%i,%i]:tensor<2x2xi"
                "32>\n%r=tensor.insert %a into %x[%i,%i]:tensor<2x2xi32>"
            ),
            "2x2xi32",
        ),
        (
            (
                "%i=arith.constant 0:index\n%a=tensor.extract %table[%i,%i]:tensor<2x2xi"
                "32>\n%r=tensor.from_elements %a:tensor<1xi32>"
            ),
            "1xi32",
        ),
        (
            '%r="tensor.concat"(%x,%table) <{dim=0:i64}> : (tensor<2x2xi32>,tensor<2x2xi32>) -> tensor<4x2xi32>',
            "4x2xi32",
        ),
        (
            (
                '%r="tensor.insert_slice"(%table,%x) <{static_offsets=array<i64:0,0>,st'
                "atic_sizes=array<i64:2,2>,static_strides=array<i64:1,1>,operandSegment"
                "Sizes=array<i32:1,1,0,0,0>}> : (tensor<2x2xi32>,tensor<2x2xi32>) -> te"
                "nsor<2x2xi32>"
            ),
            "2x2xi32",
        ),
        (
            (
                '%r="tensor.extract_slice"(%table) <{static_offsets=array<i64:0,0>,stat'
                "ic_sizes=array<i64:2,2>,static_strides=array<i64:1,1>,operandSegmentSi"
                "zes=array<i32:1,0,0,0>}> : (tensor<2x2xi32>) -> tensor<2x2xi32>"
            ),
            "2x2xi32",
        ),
        ("%r=tensor.cast %table : tensor<2x2xi32> to tensor<?x2xi32>", "?x2xi32"),
        ("%r=tensor.collapse_shape %table [[0,1]] : tensor<2x2xi32> into tensor<4xi32>", "4xi32"),
        (
            (
                "%flat=tensor.collapse_shape %table [[0,1]] : tensor<2x2xi32> into tens"
                "or<4xi32>\n%r=tensor.expand_shape %flat [[0,1]] output_shape [2,2] : te"
                "nsor<4xi32> into tensor<2x2xi32>"
            ),
            "2x2xi32",
        ),
        (
            (
                "%shape=arith.constant dense<[4]> : tensor<1xi64>\n%r=tensor.reshape %ta"
                "ble(%shape) : (tensor<2x2xi32>,tensor<1xi64>) -> tensor<4xi32>"
            ),
            "4xi32",
        ),
        (
            (
                "%e=tensor.empty() : tensor<2x2x2xi32>\n%r=linalg.broadcast ins(%table:t"
                "ensor<2x2xi32>) outs(%e:tensor<2x2x2xi32>) dimensions=[0]"
            ),
            "2x2x2xi32",
        ),
        (
            (
                "%z=arith.constant 0:i32\n%r=tensor.pad %table low[1,1] high[1,1] { ^bb0"
                "(%i:index,%j:index): tensor.yield %z:i32 } : tensor<2x2xi32> to tensor"
                "<4x4xi32>"
            ),
            "4x4xi32",
        ),
    ],
)
def test_nonuniform_literal_tables_cannot_hide_behind_movement(carrier, result_type):
    module = parse_mlir_text(
        "module { func.func @forward(%x:tensor<2x2xi32>) -> tensor<"
        + result_type
        + "> {\n%table=arith.constant dense<[[1,2],[3,4]]> : tensor<2x2xi32>\n"
        + carrier
        + "\nfunc.return %r : tensor<"
        + result_type
        + ">\n}}"
    )
    module.verify()
    # prepare() applies this same structural classifier to both source and
    # prepared IR; no source spelling or producer provenance grants an escape.
    for screened in (module, module.clone()):
        proof = construction_evidence(screened)
        assert proof["status"] == "host"
        assert any(e["reason"] == "nonuniform_literal_result_construction" for e in proof["evidence"])


def test_runtime_ieee_selection_with_literal_ancestry_is_not_an_offset():
    module = source("""%z=arith.constant 0.0 : f32
        %nan=arith.cmpf uno, %a,%a : f32
        %lhs=arith.select %nan,%z,%a : f32
        %rhs=arith.select %nan,%z,%b : f32
        %sum=arith.addf %lhs,%rhs : f32
        %i=arith.fptosi %sum : f32 to i32
        %v=arith.sitofp %i : i32 to f32""")
    assert construction_evidence(module, excluded=["readout_offset_rounding"])["status"] == "eligible"


def test_dense_splat_literal_offset_still_refuses():
    module = source(
        """%sum=arith.addf %a,%b : f32
        %i=arith.fptosi %sum : f32 to i32
        %v=arith.sitofp %i : i32 to f32""",
        literal=True,
    )
    assert construction_evidence(module, excluded=["readout_offset_rounding"])["status"] == "host"
