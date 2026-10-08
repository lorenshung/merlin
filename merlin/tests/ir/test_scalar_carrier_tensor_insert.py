"""Current packet lanes, tails, mutation refusals and ordinary native lowering."""

import ctypes
import hashlib
import json
import subprocess
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest
from xdsl.dialects import arith
from xdsl.dialects.builtin import IndexType, IntegerAttr

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.closed_tensor_insert import prove_closed_tensor_insert
from merlin.llvmlower.pipeline import apply_passes
from merlin.llvmlower.scalar_pointwise_packet import apply_for_test
from merlin.llvmlower.source_expression_interval import (
    find_closed_scalar_i8_observers,
    validate_closed_scalar_observer,
)
from merlin.llvmlower.toolchain import m2m_python

# Load the retained sibling fixture explicitly. Installed qualification uses
# Python -I and pytest importlib mode, so it must not depend on adding a test
# directory or a source checkout to sys.path.
_spec = spec_from_file_location(
    "_scalar_carrier_test_support", Path(__file__).with_name("test_source_scalar_carrier_binding.py")
)
assert _spec is not None and _spec.loader is not None
_support = module_from_spec(_spec)
_spec.loader.exec_module(_support)
EFFECTS, HOST, PROVIDER = _support.EFFECTS, _support.HOST, _support.PROVIDER
selection, source = _support.selection, _support.source


def raw_source(length=7):
    return source(length).replace(
        "%ratio=arith.divf %two,%cut",
        "%zero=arith.constant 0.0:f32\n"
        "%c1=math.fma %cut,%zero,%cut:f32\n"
        "%c2=math.fma %c1,%zero,%c1:f32\n"
        "%c3=math.fma %c2,%zero,%c2:f32\n"
        "%c4=math.fma %c3,%zero,%c3:f32\n"
        "%ratio=arith.divf %two,%c4",
    )


def packet_source(length=7, *, lanes=2, rows=None):
    raw = raw_source(length)
    if rows is not None:
        raw = (
            raw.replace(f"tensor<{length}x", f"tensor<{rows}x{length}x")
            .replace("affine_map<(d0)->(d0)>", "affine_map<(d0,d1)->(d0,d1)>")
            .replace('iterator_types=["parallel"]', 'iterator_types=["parallel","parallel"]')
        )
    fused = apply_passes(
        raw, "func.func(linalg-fuse-elementwise-ops),func.func(linalg-generalize-named-ops),canonicalize"
    )
    packet, count = apply_for_test(fused, lanes=lanes)
    assert count == 1
    # The normal current-stage transport emits generic form. xDSL's custom
    # tensor.insert parser does not accept native provenance attributes.
    return subprocess.run(
        [
            str(m2m_python()),
            "-c",
            "import sys;from torch_mlir import ir\nwith ir.Context() as ctx:\n"
            " m=ir.Module.parse(sys.stdin.read());"
            "sys.stdout.write(m.operation.get_asm(print_generic_op_form=True))",
        ],
        input=packet,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


@pytest.mark.parametrize("length,expected", [(1, 1), (2, 2), (7, 3), (16, 2)])
def test_current_packet_lane_and_tail_publications(length, expected):
    module = parse_mlir_text(packet_source(length))
    before = str(module)
    proofs, refusals = find_closed_scalar_i8_observers(module, effects=EFFECTS)
    assert len(proofs) == expected, [(op.name, reason) for op, reason in refusals]
    assert not refusals
    for proof in proofs:
        validate_closed_scalar_observer(proof)
        assert tuple(proof.integer_result.uses)[0].operation.name == "tensor.insert"
    assert str(module) == before


@pytest.mark.parametrize("lanes,length,expected", [(2, 9, 3), (4, 7, 7), (4, 12, 4)])
def test_nested_independent_tensor_shape_and_lane_width(lanes, length, expected):
    module = parse_mlir_text(packet_source(length, lanes=lanes, rows=3))
    proofs, refusals = find_closed_scalar_i8_observers(module, effects=EFFECTS)
    assert len(proofs) == expected and not refusals
    for proof in proofs:
        validate_closed_scalar_observer(proof)
        anchor = tuple(proof.integer_result.uses)[0].operation
        assert anchor.results[0].type.get_shape() == (3, length)


def test_duplicate_lane_coordinate_declines_and_stale_index_changes_refuse():
    module = parse_mlir_text(packet_source(8))
    inserts = [op for op in module.walk() if op.name == "tensor.insert"]
    assert len(inserts) == 2
    proofs, _ = find_closed_scalar_i8_observers(module, effects=EFFECTS)
    assert len(proofs) == 2
    inserts[1].operands = (*inserts[1].operands[:2], *inserts[0].operands[2:])
    with pytest.raises(ValueError, match="disjoint"):
        prove_closed_tensor_insert(inserts[0])
    for proof in proofs:
        with pytest.raises(ValueError):
            validate_closed_scalar_observer(proof)


def test_out_of_bounds_index_declines():
    module = parse_mlir_text(packet_source(8))
    inserts = [op for op in module.walk() if op.name == "tensor.insert"]
    bad = arith.ConstantOp(IntegerAttr(8, IndexType()))
    inserts[0].parent.insert_op_before(bad, inserts[0])
    inserts[0].operands = (*inserts[0].operands[:2], bad.result)
    with pytest.raises(ValueError, match="in bounds"):
        prove_closed_tensor_insert(inserts[0])


def test_all_current_packet_lanes_form_one_family_and_do_not_reparse_producers(tmp_path):
    current = packet_source(7)
    patch = selection().callback(current.encode(), tmp_path)
    assert len(patch.replacements) == 3
    record = json.loads((tmp_path / "binding.json").read_text())
    assert record["family_count"] == 1 and record["physical_table_bytes"] == 6144
    assert patch.source_sha256 == hashlib.sha256(current.encode()).hexdigest()
    assert patch.fragment.count('"memref.global"') == 1
    module = parse_mlir_text(current)
    for row in patch.replacements:
        op = module
        for region, block, ordinal in row["anchor"]:
            op = tuple(tuple(op.regions[region].blocks)[block].ops)[ordinal]
        assert op.name == "tensor.insert"
        assert all(
            item["name"] not in ("tensor.extract", "tensor.insert", "scf.for", "scf.yield") for item in row["remove"]
        )


def test_unknown_effects_decline_current_packet_lane():
    from xdsl.dialects import func
    from xdsl.dialects.builtin import i32

    module = parse_mlir_text(packet_source(8))
    insert = next(op for op in module.walk() if op.name == "tensor.insert")
    unknown = func.CallOp("unknown_environment_observer", [], [i32])
    insert.parent.insert_op_before(unknown, insert)
    proofs, refusals = find_closed_scalar_i8_observers(module, effects=EFFECTS)
    assert not proofs and len(refusals) == 2
    assert all("unknown source effects" in reason for _, reason in refusals)


def test_integer_escape_and_carrier_escape_decline():
    for integer_escape in (True, False):
        module = parse_mlir_text(packet_source(8))
        insert = next(op for op in module.walk() if op.name == "tensor.insert")
        copy = insert.clone()
        if not integer_escape:
            copy.operands = (copy.operands[0], insert.results[0], *copy.operands[2:])
        insert.parent.insert_op_after(copy, insert)
        with pytest.raises(ValueError, match="escape|observation"):
            prove_closed_tensor_insert(insert)


def test_unproved_index_arithmetic_and_empty_loop_decline():
    module = parse_mlir_text(packet_source(8))
    inserts = [op for op in module.walk() if op.name == "tensor.insert"]
    unknown_index = arith.MuliOp(inserts[0].operands[2], inserts[0].operands[2])
    inserts[0].parent.insert_op_before(unknown_index, inserts[0])
    inserts[0].operands = (*inserts[0].operands[:2], unknown_index.result)
    with pytest.raises(ValueError, match="unsupported arithmetic"):
        prove_closed_tensor_insert(inserts[0])
    module = parse_mlir_text(packet_source(8))
    insert = next(op for op in module.walk() if op.name == "tensor.insert")
    loop = insert.parent.parent_op()
    loop.operands = (loop.operands[0], loop.operands[0], *loop.operands[2:])
    with pytest.raises(ValueError, match="nonempty"):
        prove_closed_tensor_insert(insert)


def test_native_all_lane_validation_preserves_control_producers_and_destinations(tmp_path):
    from merlin.llvmlower.source_stage_transport import NATIVE

    current = packet_source(7)
    patch = selection().callback(current.encode(), tmp_path)
    source_path, packet_path = tmp_path / "current.mlir", tmp_path / "packet.json"
    source_path.write_text(current)
    packet_path.write_text(json.dumps(patch.claim()))
    script = tmp_path / "apply.py"
    script.write_text(
        "import sys,json,hashlib\nfrom torch_mlir import ir\n_run_stages=lambda *args:None\n"
        + NATIVE
        + """
with ir.Context() as ctx:
 m=ir.Module.parse(open(sys.argv[1]).read())
 original=m.operation
 before=original.get_asm(print_generic_op_form=True)
 digest=hashlib.sha256(before.encode()).hexdigest()
 def walk(op):
  yield op
  for region in op.regions:
   for block in region.blocks:
    for child in block.operations:
     yield from walk(child.operation)
 operations=list(walk(original))
 loops=[op for op in operations if op.name=='scf.for']
 inserts=[op for op in operations if op.name=='tensor.insert']
 producers=[op for op in operations if op.name=='tensor.extract']
 preserved=[(op,op.block,tuple(op.operands),tuple(op.results)) for op in [*loops,*producers]]
 publications=[(op,op.block,tuple(op.operands[1:]),tuple(op.results)) for op in inserts]
 packet=json.load(open(sys.argv[2]))
 invalid=dict(packet)
 invalid['replacements']=[*packet['replacements'],*packet['replacements']]
 try:
  _cs_apply(ctx,m,invalid,digest)
 except ValueError:
  pass
 else:
  raise AssertionError('duplicate lane publication admitted')
 assert original.get_asm(print_generic_op_form=True)==before
 _cs_apply(ctx,m,packet,digest)
 assert m.operation==original
 for op,block,operands,results in preserved:
  assert op.block==block and tuple(op.operands)==operands and tuple(op.results)==results
 for op,block,operands,results in publications:
  assert op.block==block and tuple(op.operands[1:])==operands and tuple(op.results)==results
  assert op.operands[0].owner.operation.name=='func.call'
 calls=[op for op in walk(original) if op.name=='func.call' and 'prov.scalar_carrier_source_trace' in op.attributes]
 assert len(calls)==3 and original.verify()
 print('PASS_CURRENT_LANES_CONTROL_PRODUCERS_DESTINATIONS')
"""
    )
    result = subprocess.run(
        [str(m2m_python()), str(script), str(source_path), str(packet_path)], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "PASS_CURRENT_LANES_CONTROL_PRODUCERS_DESTINATIONS"


@pytest.mark.parametrize("length", [1, 7, 16])
def test_normal_current_packet_native_tail_four_modes(tmp_path, length):
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.lower import lower_model
    from merlin.llvmlower.scalar_pointwise_packet import FEATURE
    from merlin.llvmlower.toolchain import clang

    raw = raw_source(length)
    options = dict(targets=(), features={FEATURE})
    baseline = lower_model(raw, tmp_path / "baseline", **options)
    selected = lower_model(raw, tmp_path / "selected", source_scalar_carrier=selection(), **options)
    record = selected.stats["source_scalar_carrier"]
    assert len(record["members"]) == (1 if length == 1 else 3 if length % 2 else 2)
    assert record["physical_table_bytes"] == 6144
    current = (selected.workdir / "source_scalar_carrier/current.mlir").read_text()
    assert '"tensor.insert"' in current
    if length > 1:
        assert '"scf.for"' in current
    provider = tmp_path / "provider.c"
    provider.write_text(PROVIDER)
    models = []
    for result in (baseline, selected):
        library = result.workdir / "model.so"
        subprocess.run(
            [
                str(clang()),
                "-O2",
                "-shared",
                "-fPIC",
                "-ffp-contract=off",
                str(result.ll_path),
                str(mlir_runtime_c()),
                str(provider),
                "-lm",
                "-o",
                str(library),
            ],
            check=True,
            capture_output=True,
        )
        models.append(HostModel.load(str(library)))
    values = np.linspace(1.125, 1.875, length, dtype=np.float32)
    upstream = np.linspace(-19.127, 17.129, length, dtype=np.float32)
    original = values.tobytes(), upstream.tobytes()
    environment = (ctypes.c_longdouble * 32)()
    assert HOST.fegetenv(ctypes.byref(environment)) == 0
    try:
        for mode in (0, 0x400, 0x800, 0xC00):
            assert HOST.fesetround(mode) == 0
            outputs = []
            for model in models:
                out = np.full(length + 32, 109, dtype=np.int8)
                model(
                    [
                        (values.ctypes.data, values.shape),
                        (upstream.ctypes.data, upstream.shape),
                        (out.ctypes.data + 16, values.shape),
                    ]
                )
                assert HOST.fegetround() == mode
                assert np.all(out[:16] == 109) and np.all(out[-16:] == 109)
                outputs.append(out[16:-16].copy())
            np.testing.assert_array_equal(*outputs)
    finally:
        assert HOST.fesetenv(ctypes.byref(environment)) == 0
    assert (values.tobytes(), upstream.tobytes()) == original
