"""Execute generated host limb arithmetic at native wraparound boundaries."""

import json

import numpy as np
import pytest
from xdsl.dialects.builtin import IntegerType

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.boolean_components import rewrite_boolean_components
from merlin.llvmlower.integer_components import rewrite_integer_components


def module(dtype, expression):
    return parse_mlir_text(
        """builtin.module {
    func.func @forward(%x: tensor<4xTYPE>, %y: tensor<4xTYPE>) -> tensor<4xTYPE> {
      %empty = tensor.empty() : tensor<4xTYPE>
      %result = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]
      } ins(%x, %y : tensor<4xTYPE>, tensor<4xTYPE>) outs(%empty : tensor<4xTYPE>) {
      ^bb0(%a: TYPE, %b: TYPE, %c: TYPE):
        EXPR
        linalg.yield %v : TYPE
      } -> tensor<4xTYPE>
      func.return %result : tensor<4xTYPE>
    }}""".replace("TYPE", dtype).replace("EXPR", expression)
    )


def evaluate(module, arrays):
    """Small integer SSA interpreter; all generated tensors have identity maps."""
    function = next(op for op in module.walk() if op.name == "func.func")
    values = dict(zip(function.body.block.args, [np.array(a, dtype=object) for a in arrays]))

    def normalize(value, typ):
        if isinstance(typ, IntegerType):
            bits = typ.width.data
            value = int(value) % (1 << bits)
            return value - (1 << bits) if value >= (1 << (bits - 1)) else value
        return value

    def scalar(op, env):
        a = [env[v] for v in op.operands]
        name = op.name
        if name == "arith.constant":
            result = op.value.value.data
            if not isinstance(op.results[0].type, IntegerType):
                result = int.from_bytes(op.results[0].type.pack([result]), "little")
        elif name in {"arith.extsi", "arith.trunci", "arith.bitcast"}:
            result = a[0]
        elif name == "arith.extui":
            result = a[0] % (1 << op.operands[0].type.width.data)
        elif name == "arith.addi":
            result = a[0] + a[1]
        elif name == "arith.subi":
            result = a[0] - a[1]
        elif name == "arith.andi":
            result = a[0] & a[1]
        elif name == "arith.muli":
            result = a[0] * a[1]
        elif name == "arith.shrsi":
            result = a[0] >> a[1]
        elif name == "arith.shli":
            assert 0 <= a[1] < op.results[0].type.width.data
            result = a[0] << a[1]
        elif name == "arith.cmpi":
            predicate = op.predicate.value.data
            assert predicate in (1, 2, 8)
            result = int(
                a[0] != a[1]
                if predicate == 1
                else a[0] < a[1]
                if predicate == 2
                else a[0] % (1 << op.operands[0].type.width.data) > a[1] % (1 << op.operands[1].type.width.data)
            )
        elif name == "arith.cmpf":
            import struct

            assert str(op.operands[0].type) == "f16" and op.predicate.value.data == 4
            left, right = [struct.unpack("<e", (int(value) & 0xFFFF).to_bytes(2, "little"))[0] for value in a]
            result = int(left < right)
        elif name == "arith.select":
            result = a[1] if a[0] else a[2]
        else:
            raise AssertionError(name)
        env[op.results[0]] = normalize(result, op.results[0].type)

    for op in function.body.block.ops:
        if op.name == "func.return":
            return values[op.operands[0]].tolist()
        if op.name == "arith.constant":
            scalar(op, values)
        elif op.name == "tensor.empty":
            values[op.results[0]] = np.zeros(op.results[0].type.get_shape(), object)
        elif op.name == "tensor.splat":
            values[op.results[0]] = np.full(op.results[0].type.get_shape(), values[op.operands[0]], object)
        elif op.name == "linalg.generic":
            from xdsl.ir.affine import AffineDimExpr

            output = values[op.outputs[0]].copy()
            maps = [attr.data for attr in op.get_indexing_maps()]
            extents = [None] * len(op.get_iterator_types())
            for value, amap in zip(op.operands, maps):
                for extent, expr in zip(value.type.get_shape(), amap.results):
                    assert isinstance(expr, AffineDimExpr)
                    previous = extents[expr.position]
                    assert previous is None or previous == extent
                    extents[expr.position] = extent
            assert all(n is not None for n in extents)
            for index in np.ndindex(tuple(extents)):
                output_index = tuple(maps[-1].eval(index, ()))
                env = {
                    arg: (
                        output[output_index] if i == len(op.inputs) else values[source][tuple(maps[i].eval(index, ()))]
                    )
                    for i, (arg, source) in enumerate(zip(op.body.block.args, op.operands))
                }
                for child in op.body.block.ops:
                    if child.name == "linalg.yield":
                        output[output_index] = env[child.operands[0]]
                    else:
                        scalar(child, env)
            values[op.results[0]] = output
        else:
            raise AssertionError(op.name)
    raise AssertionError("no return")


@pytest.mark.parametrize("bits", [8, 32, 64])
@pytest.mark.parametrize("coefficient", [-3, 1, 7])
def test_linear_components_recombine_with_native_wraparound(bits, coefficient):
    source = module(
        f"i{bits}",
        f"%s = arith.constant {coefficient} : i{bits}\n"
        f"%p = arith.muli %a, %s : i{bits}\n%v = arith.addi %p, %b : i{bits}",
    )
    limit = 1 << (bits - 1)
    x, y = [-limit, limit - 1, -129 % limit, 128 % limit], [1, -1, 127, -128]
    plans = rewrite_integer_components(source, limb_attribute="test.limb", input_bounds=[limit, 128])
    source.verify()
    assert plans and all(p["radix"] == 256 for p in plans)
    assert all(
        json.loads(op.attributes["test.limb"].data)["family"] == "F2"
        for op in source.walk()
        if "test.limb" in op.attributes
    )
    expected = [((coefficient * a + b + limit) % (2 * limit)) - limit for a, b in zip(x, y)]
    assert evaluate(source, [x, y]) == expected


def test_boolean_or_stays_in_the_host_source():
    source = module("i1", "%v = arith.ori %a, %b : i1")
    original = str(source)
    bounds, count = rewrite_boolean_components(source)
    assert count == 0 and bounds == {}
    assert str(source) == original
    assert not rewrite_integer_components(source, limb_attribute="test.limb", input_bounds=[1, 1], known_bounds=bounds)
    assert str(source) == original


def test_widening_keeps_the_carry_above_the_original_storage_width():
    source = parse_mlir_text("""builtin.module {
      func.func @forward(%x: tensor<4xi64>, %unused: tensor<4xi64>) -> tensor<4xi128> {
        %e = tensor.empty() : tensor<4xi128>
        %r = linalg.generic {
          indexing_maps = [affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],
          iterator_types = ["parallel"]
        } ins(%x : tensor<4xi64>) outs(%e : tensor<4xi128>) {
        ^bb0(%a: i64, %b: i128):
          %v = arith.extsi %a : i64 to i128
          linalg.yield %v : i128
        } -> tensor<4xi128>
        func.return %r : tensor<4xi128>
      }}""")
    values = [(1 << 63) - 1, -(1 << 63), 32767, -32768]
    plans = rewrite_integer_components(source, limb_attribute="test.limb", input_bounds=[1 << 63, 0])
    assert len(plans) == 8  # Source-width storage is normalized before the signed extension.
    assert evaluate(source, [values, [0] * 4]) == values


def test_native_sign_transform_keeps_all_ieee_payload_bits_in_integer_literals():
    from merlin.llvmlower.native_sign_components import rewrite_native_sign_components

    source = parse_mlir_text("""builtin.module {
      func.func @forward(%x: tensor<4xf16>) -> tensor<4xf16> {
        %e = tensor.empty() : tensor<4xf16>
        %r = linalg.generic {
          indexing_maps = [affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],
          iterator_types = ["parallel"]
        } ins(%x : tensor<4xf16>) outs(%e : tensor<4xf16>) {
        ^bb0(%a: f16, %b: f16):
          %v = arith.negf %a : f16
          linalg.yield %v : f16
        } -> tensor<4xf16>
        func.return %r : tensor<4xf16>
      }}""")
    bounds, proofs = rewrite_native_sign_components(source)
    assert proofs == [
        dict(
            source="arith.negf",
            representation="ieee_storage_bits",
            width=16,
            arithmetic="sign_bit_offset_modulo_storage_width",
            nan_payload="preserved",
        )
    ]
    assert not any(op.name == "arith.negf" for op in source.walk())
    assert rewrite_integer_components(source, limb_attribute="test.limb", input_bounds=[0], known_bounds=bounds)
    source.verify()
    for words in [[0, 0x8000, 0x7C00, 0xFC00], [0x7E01, 0xFE7F, 0x7C01, 0xFC01], [0x0001, 0x8001, 0x3C00, 0xBC00]]:
        assert [value & 0xFFFF for value in evaluate(source, [words])] == [word ^ 0x8000 for word in words]


@pytest.mark.parametrize("dtype,predicate,zero", [("f16", "olt", "0.0"), ("i64", "slt", "0")])
def test_clamp_epilogue_preserves_source_comparison_and_payloads(dtype, predicate, zero):
    from merlin.llvmlower.native_sign_components import rewrite_native_sign_components

    source = parse_mlir_text(
        """builtin.module {
      func.func @forward(%x: tensor<4xTYPE>) -> tensor<4xTYPE> {
        %e = tensor.empty() : tensor<4xTYPE>
        %r = linalg.generic {
          indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>], iterator_types=["parallel"]
        } ins(%x : tensor<4xTYPE>) outs(%e : tensor<4xTYPE>) {
        ^bb0(%a: TYPE,%unused: TYPE):
          %zero=arith.constant ZERO : TYPE
          %p=arith.CMP PRED,%a,%zero : TYPE
          %v=arith.select %p,%zero,%a : TYPE
          linalg.yield %v : TYPE
        } -> tensor<4xTYPE>
        func.return %r : tensor<4xTYPE>
      }}""".replace("TYPE", dtype)
        .replace("ZERO", zero)
        .replace("PRED", predicate)
        .replace("CMP", "cmpf" if dtype == "f16" else "cmpi")
    )
    bounds, proofs = rewrite_native_sign_components(source)
    assert proofs[0]["arithmetic"] == "source_compare_select_retained_host"
    plans = rewrite_integer_components(source, limb_attribute="test.limb", input_bounds=[1 << 63], known_bounds=bounds)
    assert plans
    source.verify()
    if dtype == "f16":
        for words, expected in [
            ([0, 0x8000, 0x7C00, 0xFC00], [0, 0x8000, 0x7C00, 0]),
            ([0x7E01, 0xFE7F, 0x7C01, 0xFC01], [0x7E01, 0xFE7F, 0x7C01, 0xFC01]),
            ([0x0001, 0x8001, 0x3C00, 0xBC00], [0x0001, 0, 0x3C00, 0]),
        ]:
            assert [v & 0xFFFF for v in evaluate(source, [words])] == expected
    else:
        assert evaluate(source, [[-(1 << 63), -1, 0, (1 << 63) - 1]]) == [0, 0, 0, (1 << 63) - 1]


def test_window_reduction_ignores_unread_shape_operand():
    source = parse_mlir_text("""module {
      func.func @forward(%x: tensor<4xi64>) -> tensor<2xi64> {
        %shape = tensor.empty() : tensor<2xi64>
        %z = arith.constant 0 : i64
        %seed = tensor.splat %z : tensor<2xi64>
        %r = linalg.generic {
          indexing_maps = [affine_map<(d0,d1)->(d0*2+d1)>, affine_map<(d0,d1)->(d1)>, affine_map<(d0,d1)->(d0)>],
          iterator_types = ["parallel", "reduction"]
        } ins(%x, %shape : tensor<4xi64>, tensor<2xi64>) outs(%seed : tensor<2xi64>) {
        ^bb0(%a: i64, %unused: i64, %acc: i64):
          %v = arith.addi %a, %acc : i64
          linalg.yield %v : i64
        } -> tensor<2xi64>
        return %r : tensor<2xi64>
      }
    }""")
    plans = rewrite_integer_components(source, limb_attribute="test.limb", input_bounds=[128])
    assert plans
    source.verify()
    for op in source.walk():
        if "test.limb" in op.attributes:
            assert len(op.inputs) == 2
            assert len(op.get_indexing_maps()) == 3
            assert not op.body.block.args[1].uses
            plan = json.loads(op.attributes["test.limb"].data)
            assert {entry["operand"] for entry in plan["inputs"]} == {0, 1}
            data = next(entry for entry in plan["inputs"] if entry["operand"] == 0)
            assert next(entry for entry in plan["inputs"] if entry["operand"] == 1) == dict(
                operand=1, index=0, count=1, exponent=8 * data["index"] + data["exponent"]
            )


def test_declared_bfloat_nan_negation_uses_device_words_then_cpu_canonicalization():
    from merlin.llvmlower.native_sign_components import rewrite_native_sign_components

    source = parse_mlir_text("""module {
      func.func @forward(%x: tensor<4xbf16>) -> tensor<4xbf16> {
        %e=tensor.empty() : tensor<4xbf16>
        %r=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
          ins(%x : tensor<4xbf16>) outs(%e : tensor<4xbf16>) {
        ^bb0(%a:bf16,%b:bf16):
          %v=arith.negf %a : bf16
          linalg.yield %v : bf16
        } -> tensor<4xbf16>
        return %r : tensor<4xbf16>
      }}""")
    bounds, proofs = rewrite_native_sign_components(source, canonical_bf16_nan=True)
    assert proofs[0]["nan_payload"] == "canonical_positive_quiet"
    assert rewrite_integer_components(source, limb_attribute="test.limb", input_bounds=[0], known_bounds=bounds)
    for payload in range(1, 128):
        words = [0x7F80 | payload, 0xFF80 | payload, 0, 0x8000]
        assert [value & 0xFFFF for value in evaluate(source, [words])] == [0x7FC0, 0x7FC0, 0x8000, 0]


def test_binary32_sign_words_preserve_nonfinite_payloads_without_quantization():
    from merlin.llvmlower.native_sign_components import rewrite_native_sign_components

    source = parse_mlir_text("""module {
      func.func @forward(%x:tensor<4xf32>) -> tensor<4xf32> {
        %e=tensor.empty() : tensor<4xf32>
        %r=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
          ins(%x:tensor<4xf32>) outs(%e:tensor<4xf32>) {
        ^bb0(%a:f32,%unused:f32):
          %v=arith.negf %a : f32
          linalg.yield %v : f32
        } -> tensor<4xf32>
        return %r : tensor<4xf32>
      }}""")
    bounds, proofs = rewrite_native_sign_components(source)
    assert proofs[0]["nan_payload"] == "preserved"
    assert rewrite_integer_components(source, limb_attribute="test.limb", input_bounds=[0], known_bounds=bounds)
    for words in [
        [0, 0x80000000, 0x7F800000, 0xFF800000],
        [0x7FC00001, 0xFFC00012, 0x7F800001, 0xFF800001],
        [1, 0x80000001, 0x7F7FFFFF, 0xFF7FFFFF],
    ]:
        assert [v & 0xFFFFFFFF for v in evaluate(source, [words])] == [v ^ 0x80000000 for v in words]


@pytest.mark.parametrize("bits", [8, 16, 32])
@pytest.mark.parametrize("cast", ["extui", "extsi"])
@pytest.mark.parametrize("wrap", [False, True])
def test_width_changes_observe_the_source_ring(bits, cast, wrap):
    limit = 1 << (bits - 1)
    body = f"%s=arith.addi %a,%b : i{bits}\n" if wrap else ""
    body += f"%v=arith.{cast} " + ("%s" if wrap else "%a") + f" : i{bits} to i64"
    source = parse_mlir_text(
        """module {
      func.func @forward(%x:tensor<4xI>,%y:tensor<4xI>) -> tensor<4xi64> {
        %e=tensor.empty() : tensor<4xi64>
        %r=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,
                      affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
        ins(%x,%y:tensor<4xI>,tensor<4xI>) outs(%e:tensor<4xi64>) {
          ^bb0(%a:I,%b:I,%unused:i64):
          BODY
          linalg.yield %v : i64
        } -> tensor<4xi64>
        func.return %r : tensor<4xi64>
      }}""".replace("BODY", body).replace("I", f"i{bits}")
    )
    values = [-1, -limit, 0, limit - 1]
    other = [1, -1, limit - 1, 1]
    expected = evaluate(source, [values, other])
    plans = rewrite_integer_components(source, limb_attribute="test.limb", input_bounds=[limit, limit])
    assert plans
    assert evaluate(source, [values, other]) == expected
    if not wrap and cast == "extui":
        assert expected == [(v % (1 << bits)) for v in values]
    if wrap and cast == "extsi":
        assert expected[-1] == -limit


@pytest.mark.parametrize("bits", [8, 16, 32])
def test_truncation_before_signed_extension_keeps_its_modular_boundary(bits):
    text = str(module("i64", f"%t=arith.trunci %a : i64 to i{bits}\n%v=arith.extsi %t : i{bits} to i64")).replace(
        "iBITS", f"i{bits}"
    )
    source = parse_mlir_text(text)
    values = [-(1 << 63), (1 << bits) - 1, (1 << (bits - 1)), (1 << (bits - 1)) - 1]
    expected = evaluate(source, [values, [0] * 4])
    assert rewrite_integer_components(source, limb_attribute="test.limb", input_bounds=[1 << 63, 0])
    assert evaluate(source, [values, [0] * 4]) == expected


def test_real_frontend_uint8_cast_review_repro():
    torch = pytest.importorskip("torch")
    m2m = pytest.importorskip("m2m")

    class Cast(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten._to_copy.default(x, dtype=torch.int32)

    values = torch.tensor([255, 128, 0, 127], dtype=torch.uint8)
    original = parse_mlir_text(str(m2m.convert(Cast(), (values,), backend="fx_importer", fully_standard=True).module))
    assert rewrite_integer_components(original, limb_attribute="test.limb", input_bounds=[128])
    assert evaluate(original, [[-1, -128, 0, 127]]) == [255, 128, 0, 127]


def test_late_width_pass_does_not_split_generated_quotient_plumbing():
    from merlin.llvmlower.integer_width_boundaries import preserve_integer_width_boundaries

    source = module(
        "i64", "%q=arith.divsi %a,%b : i64\n%t=arith.trunci %q : i64 to i32\n%v=arith.extsi %t : i32 to i64"
    )
    before = str(source)
    assert preserve_integer_width_boundaries(source) == 0
    assert str(source) == before


@pytest.mark.parametrize("coefficient", [(1 << 40) + 3, -(1 << 63) + 1, (1 << 63) - 1])
def test_full_width_scalar_coefficients_use_exact_literal_planes(coefficient):
    source = module("i64", f"%s=arith.constant {coefficient}:i64\n%v=arith.muli %a,%s:i64")
    limit = 1 << 63
    values = [-limit, limit - 1, -129, 128]
    plans = rewrite_integer_components(source, limb_attribute="test.limb", input_bounds=[limit, limit])
    assert plans
    expected = [((coefficient * a + limit) % (2 * limit)) - limit for a in values]
    assert evaluate(source, [values, [0] * 4]) == expected


@pytest.mark.parametrize("bits", [8, 32, 64])
def test_named_sum_exposes_full_width_limb_reduction_and_modular_wrap(bits):
    source = parse_mlir_text(
        """module {
      func.func @forward(%x:tensor<3x4xTYPE>) -> tensor<4xTYPE> {
        %z=arith.constant 0:TYPE
        %s=tensor.splat %z:tensor<4xTYPE>
        %r=linalg.reduce ins(%x:tensor<3x4xTYPE>) outs(%s:tensor<4xTYPE>) dimensions=[0]
          (%a:TYPE,%b:TYPE) { %v=arith.addi %a,%b:TYPE
            linalg.yield %v:TYPE }
        return %r:tensor<4xTYPE>
      }}""".replace("TYPE", f"i{bits}")
    )
    limit = 1 << (bits - 1)
    values = [[limit - 1, -limit, 127, -128]] * 3
    plans = rewrite_integer_components(source, limb_attribute="test.limb", input_bounds=[limit])
    assert plans and not any(op.name == "linalg.reduce" for op in source.walk())
    assert max(p["inputs"][0]["count"] for p in plans) == bits // 8
    assert evaluate(source, [values]) == [((3 * a + limit) % (2 * limit)) - limit for a in values[0]]
