"""Actual encoded activation instructions must satisfy the raw-readout contract."""

import pytest

from merlin.targetgen.native_byte_activation import audit_program


def program():
    weights = "".join("\\01" if i == j else "\\00" for i in range(16) for j in range(16))
    instructions = [
        (0, (0x3F800000 << 32) | 9, 16),
        (0, (0x3F800000 << 32) | 1, 16),
        (0, 2, (0x3F800000 << 32) | 64),
        (1, "ptrtoint (ptr @unit__relu_identity to i64)", (16 << 48) | (16 << 32)),
        (2, "%input", (1 << 48) | (16 << 32) | 32),
        (0, (0x3F800000 << 32) | (1 << 16) | 12, 1 << 48),
        (4, (1 << 48) | (16 << 32) | 32, (16 << 48) | (16 << 32) | 0xFFFFFFFF),
        (0, (0x3F800000 << 32) | (1 << 16) | 4, 1 << 48),
        (4, (1 << 48) | (16 << 32) | 64, (16 << 48) | (16 << 32) | 0xFFFFFFFF),
        (3, 1234, (1 << 48) | (16 << 32) | 0xA0000000),
    ]
    code = "\n".join(
        f'call void asm sideeffect ".insn r 0x7b, 0x3, 0x{f:x}, x0, $0, $1", "r,r,~{{memory}}"(i64 {a}, i64 {b})'
        for f, a, b in instructions
    )
    return (
        f'@unit__relu_identity = internal constant [256 x i8] c"{weights}"\ndefine void @unit(ptr %x) {{\n{code}\n}}\n'
    )


def test_zero_shift_signed_byte_relu_and_raw_widening_are_authenticated():
    proof = audit_program(program(), "unit")
    assert proof["activated"] == proof["unactivated"] == proof["readouts"] == 1
    assert proof["computes"] == 2 and proof["mesh_partial_absolute_bound"] < 2**19


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.replace("\\01", "\\02", 1),
        lambda s: s.replace(str((0x3F800000 << 32) | (1 << 16) | 12), str((0x3F800000 << 32) | (1 << 16) | 8)),
        lambda s: s.replace(str((0x3F800000 << 32) | (1 << 16) | 12), str((0x40000000 << 32) | (1 << 16) | 12)),
        lambda s: s.replace(str((16 << 48) | (16 << 32) | 0xFFFFFFFF), str((16 << 48) | (16 << 32) | 0x80000000)),
        lambda s: s.replace(str((1 << 48) | (16 << 32) | 0xA0000000), str((1 << 48) | (16 << 32) | 0x80000000)),
        lambda s: s.replace(str((1 << 48) | (16 << 32) | 32), str((1 << 48) | (17 << 32) | 32)),
        lambda s: s.replace("0x4, x0", "0x5, x0", 1),
    ],
)
def test_changed_weights_os_scales_mesh_d_or_narrow_readout_refuse(mutate):
    with pytest.raises(ValueError):
        audit_program(mutate(program()), "unit")


@pytest.mark.parametrize(
    "dtype,zero,predicate",
    [("i8", "0", "slt"), ("i64", "0", "slt"), ("f16", "0.0", "olt"), ("bf16", "0.0", "olt"), ("f32", "0.0", "olt")],
)
def test_a_native_zero_constant_needs_no_intermediate_integer_cast(dtype, zero, predicate):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects.lowering.contraction_coverage import is_compare_select_relu

    compare = "cmpi" if dtype.startswith("i") else "cmpf"
    module = parse_mlir_text(f"""module {{func.func @forward(%x:tensor<4x{dtype}>) -> tensor<4x{dtype}> {{
      %e=tensor.empty() : tensor<4x{dtype}>
      %r=linalg.generic {{indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}}
        ins(%x:tensor<4x{dtype}>) outs(%e:tensor<4x{dtype}>) {{
        ^bb0(%a:{dtype},%out:{dtype}):
        %zero=arith.constant {zero} : {dtype}
        %p=arith.{compare} {predicate},%a,%zero : {dtype}
        %v=arith.select %p,%zero,%a : {dtype}
        linalg.yield %v : {dtype}
      }} -> tensor<4x{dtype}>
      return %r : tensor<4x{dtype}>
    }} }}""")
    root = next(op for op in module.walk() if op.name == "linalg.generic")
    assert is_compare_select_relu(root)
    constant = next(op for op in root.body.block.ops if op.name == "arith.constant")
    from xdsl.dialects.builtin import FloatAttr, IntegerAttr

    element = root.body.block.args[0].type
    constant.properties["value"] = IntegerAttr(1, element) if dtype.startswith("i") else FloatAttr(1.0, element)
    assert not is_compare_select_relu(root)
