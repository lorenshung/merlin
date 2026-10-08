"""Current native composition, independent tails and fail-before-edit contracts."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import struct
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from merlin.llvmlower.source_expression_interval import IntervalEffectContract
from merlin.llvmlower.source_scalar_carrier_binding import (
    CompilerHostNumericAdmission,
    IncomingRNECapability,
    SourceScalarCarrierSelection,
    host_admitted,
)
from merlin.llvmlower.source_scalar_carrier_policy import ApproximateScalarCarrierPolicy, ScalarCarrierErrorBudget
from merlin.llvmlower.source_stage_transport import NATIVE, ScalarLeafPatch, insert_marker, run_command

EFFECTS = IntervalEffectContract(True, True, True, True, True)
PROVIDER = "#include <fenv.h>\n_Bool current_rne(void){return fegetround()==FE_TONEAREST;}\n"
IDENTITY = hashlib.sha256(PROVIDER.encode()).hexdigest()
HOST = ctypes.CDLL(None)
HOST.fegetround.restype = ctypes.c_int
HOST.fesetround.argtypes = [ctypes.c_int]
HOST.fegetenv.argtypes = [ctypes.c_void_p]
HOST.fesetenv.argtypes = [ctypes.c_void_p]


@contextmanager
def host_rne():
    # Qualification provider's opaque aligned capacity; not a core ABI fact.
    environment = (ctypes.c_longdouble * 32)()
    assert HOST.fegetenv(ctypes.byref(environment)) == 0
    try:
        assert HOST.fesetround(0) == 0
        yield
    finally:
        assert HOST.fesetenv(ctypes.byref(environment)) == 0


def word(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def proposal(expression, effects, policy):
    # A proposal only. The current expression's full subcell proof admits or
    # invalidates it independently; expression names/labels select nothing.
    rows = [(word(float("nan")), 0, 0)] * (1 << policy.leading_bits)
    rows[word(1.0) >> (32 - policy.leading_bits)] = (word(2.0), 0, 0)
    return tuple(rows)


def selection(**changes):
    return replace(
        SourceScalarCarrierSelection(
            ApproximateScalarCarrierPolicy(
                ScalarCarrierErrorBudget(0.5, 0.01), 9, 12, 6144, 32768, True, True, True, True
            ),
            EFFECTS,
            proposal,
            IncomingRNECapability("current_rne", IDENTITY, True, True, True, True, True),
            CompilerHostNumericAdmission(
                host_rne, lambda: HOST.fegetround() == 0, hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            ),
            6144,
        ),
        **changes,
    )


def quantizer():
    return """%prod=arith.mulf %act,%up:f32
      %factor=arith.constant 3.0:f32
      %scaled=arith.mulf %prod,%factor:f32
      %lo=arith.constant -128.0:f32
      %hi=arith.constant 127.0:f32
      %a=arith.maximumf %scaled,%lo:f32
      %b=arith.minimumf %a,%hi:f32
      %t=arith.fptosi %b:f32 to i8
      %tf=arith.sitofp %t:i8 to f32
      %frac=arith.subf %b,%tf:f32
      %neg=arith.negf %frac:f32
      %abs=arith.maximumf %frac,%neg:f32
      %half=arith.constant 0.5:f32
      %gt=arith.cmpf ogt,%abs,%half:f32
      %eq=arith.cmpf oeq,%abs,%half:f32
      %one=arith.constant 1:i8
      %zero=arith.constant 0:i8
      %odd=arith.andi %t,%one:i8
      %ne=arith.cmpi ne,%odd,%zero:i8
      %tie=arith.andi %eq,%ne:i1
      %bump=arith.ori %gt,%tie:i1
      %zf=arith.constant 0.0:f32
      %negative=arith.cmpf olt,%b,%zf:f32
      %mone=arith.constant -1:i8
      %sign=arith.select %negative,%mone,%one:i8
      %delta=arith.select %bump,%sign,%zero:i8
      %r=arith.addi %t,%delta:i8"""


def source(length=7):
    return f"""module attributes {{prov.import = "retained-import"}} {{
    func.func @forward(%x:tensor<{length}xf32>,%u:tensor<{length}xf32>)->tensor<{length}xi8>
      attributes {{prov.source_node_ids=["arbitrary-parent"]}} {{
      %e=tensor.empty():tensor<{length}xf32>
      %activated_tensor=linalg.generic {{indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],
        iterator_types=["parallel"]}}
        ins(%x:tensor<{length}xf32>) outs(%e:tensor<{length}xf32>) {{
        ^bb0(%cut:f32,%unused:f32):
          %two=arith.constant 2.0:f32
          %ratio=arith.divf %two,%cut {{prov.source_node_ids=["division-join"]}}:f32
          %act=arith.mulf %cut,%ratio {{prov.source_node_ids=["activation-join"]}}:f32
          linalg.yield %act:f32
      }}->tensor<{length}xf32>
      %o=tensor.empty():tensor<{length}xi8>
      %result_tensor=linalg.generic {{indexing_maps=[affine_map<(d0)->(d0)>,
        affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],
        iterator_types=["parallel"]}}
        ins(%activated_tensor,%u:tensor<{length}xf32>,tensor<{length}xf32>) outs(%o:tensor<{length}xi8>) {{
        ^bb0(%act:f32,%up:f32,%unused:i8):
          {quantizer()}
          linalg.yield %r:i8
      }}->tensor<{length}xi8>
      return %result_tensor:tensor<{length}xi8>
    }} }}"""


@pytest.mark.parametrize("missing", tuple(vars(EFFECTS)))
def test_every_source_permission_refuses_before_start(missing):
    with pytest.raises(ValueError):
        selection(effects=replace(EFFECTS, **{missing: False})).validate()


@pytest.mark.parametrize(
    "missing",
    ("reads_actual_incoming_rounding", "preserves_rounding_mode", "preserves_flags", "nontrapping", "no_memory_writes"),
)
def test_each_runtime_permission_remains_explicit(missing):
    with pytest.raises(ValueError):
        selection(incoming_rne=replace(selection().incoming_rne, **{missing: False})).validate()


def test_hoisted_or_false_host_capability_refuses():
    with pytest.raises(ValueError):
        selection(incoming_rne=replace(selection().incoming_rne, placement="per_observer")).validate()
    capability = replace(selection().compiler_host, verify_rne=lambda: False)
    with pytest.raises(ValueError):
        with capability.enter():
            raise AssertionError("refused scope entered")


def test_positional_selection_receives_same_compiler_host_admission():
    @host_admitted
    def selected(source_scalar_carrier=None):
        return HOST.fegetround()

    incoming = HOST.fegetround()
    try:
        assert HOST.fesetround(0x800) == 0
        assert selected(selection()) == 0 and HOST.fegetround() == 0x800
        assert selected(source_scalar_carrier=selection()) == 0 and HOST.fegetround() == 0x800
        assert selected() == 0x800
    finally:
        assert HOST.fesetround(incoming) == 0


def test_marker_requires_normal_completed_fusion_and_one_bufferization():
    ordinary = (
        "canonicalize,func.func(linalg-fuse-elementwise-ops),"
        "func.func(linalg-generalize-named-ops),one-shot-bufferize,convert-linalg-to-loops"
    )
    result = insert_marker(ordinary)
    assert result.index("__merlin_current_scalar_leaf_rewrite__") < result.index("one-shot-bufferize")
    for pipeline in (
        result,
        "one-shot-bufferize",
        ordinary + ",one-shot-bufferize",
        ordinary + ",func.func(linalg-generalize-named-ops)",
    ):
        with pytest.raises(ValueError):
            insert_marker(pipeline)


def test_provenance_stripping_cannot_precede_current_source_binding(tmp_path):
    from merlin.llvmlower.lower import lower_model
    from merlin.llvmlower.prov_cse import FEATURE

    with pytest.raises(ValueError, match="provenance-stripping"):
        lower_model(source(), tmp_path, targets=(), features={FEATURE}, source_scalar_carrier=selection())
    assert not any(tmp_path.iterdir())


def test_packets_are_one_shot_and_refuse_foreign_resources():
    packet = ScalarLeafPatch("a" * 64, "module {}", ({"test": "private"},), {})
    packet.claim()
    with pytest.raises(ValueError):
        packet.claim()
    with pytest.raises(ValueError):
        ScalarLeafPatch("a" * 64, "dense_resource", ({},), {}).claim()


def test_callback_bound_to_current_source_preserves_all_inputs(tmp_path):
    from merlin.llvmlower.pipeline import apply_passes

    raw = source()
    current = apply_passes(
        raw, "func.func(linalg-fuse-elementwise-ops),func.func(linalg-generalize-named-ops),canonicalize"
    )
    patch = selection().callback(current.encode(), tmp_path)
    assert patch.source_sha256 == hashlib.sha256(current.encode()).hexdigest()
    assert len(patch.replacements) == 1
    assert patch.replacements[0]["cut"]["type"] == "f32"
    assert patch.replacements[0]["up"]["type"] == "f32"
    assert patch.fragment.count('"memref.global"') == 1
    assert "division-join" in patch.fragment and "activation-join" in patch.fragment


def test_callback_exception_terminates_only_owned_child(tmp_path):
    child = tmp_path / "child.py"
    child.write_text(
        "import hashlib,json,pathlib,time\n"
        "p=pathlib.Path(__file__).parent\nb=b'current'\n(p/'current.mlir').write_bytes(b)\n"
        "(p/'request.json').write_text(json.dumps({\n"
        " 'schema':'merlin.current_scalar_stage_request.v1',\n"
        " 'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}))\ntime.sleep(60)\n"
    )

    def fail(payload, directory):
        raise ValueError("explicit callback refusal")

    with pytest.raises(ValueError, match="explicit callback refusal"):
        run_command(
            [sys.executable, str(child)],
            directory=tmp_path,
            callback=fail,
            max_source_bytes=1024,
            max_response_bytes=1024,
            timeout=5,
        )
    assert not (tmp_path / "response.json").exists()


def test_no_request_timeout_restores_host_environment_and_only_owned_child(tmp_path):
    HOST.fetestexcept.argtypes = [ctypes.c_int]
    HOST.feraiseexcept.argtypes = [ctypes.c_int]
    HOST.feclearexcept.argtypes = [ctypes.c_int]
    incoming = HOST.fegetround()
    assert HOST.fesetround(0x800) == 0
    assert HOST.feraiseexcept(0x20) == 0
    flags = HOST.fetestexcept(0x3F)
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            with selection().compiler_host.enter():
                assert HOST.fegetround() == 0
                assert HOST.feclearexcept(0x3F) == 0
                run_command(
                    [sys.executable, "-c", "import time;time.sleep(60)"],
                    directory=tmp_path,
                    callback=lambda *args: None,
                    max_source_bytes=1024,
                    max_response_bytes=1024,
                    timeout=0.1,
                )
        assert HOST.fegetround() == 0x800 and HOST.fetestexcept(0x3F) == flags
    finally:
        assert HOST.fesetround(incoming) == 0


def test_foreign_source_and_private_response_quotas_refuse(tmp_path):
    child = tmp_path / "child.py"
    child.write_text(
        "import hashlib,json,pathlib,time\n"
        "p=pathlib.Path(__file__).parent\nb=b'current'\n(p/'current.mlir').write_bytes(b)\n"
        "(p/'request.json').write_text(json.dumps({\n"
        " 'schema':'merlin.current_scalar_stage_request.v1',\n"
        " 'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}))\ntime.sleep(60)\n"
    )
    with pytest.raises(ValueError, match="stale"):
        run_command(
            [sys.executable, str(child)],
            directory=tmp_path,
            callback=lambda *args: ScalarLeafPatch("b" * 64, "module {}", ({},), {}),
            max_source_bytes=1024,
            max_response_bytes=1024,
            timeout=5,
        )
    assert not (tmp_path / "response.json").exists()


def test_response_budget_refuses_without_publication(tmp_path):
    child = tmp_path / "child.py"
    child.write_text(
        "import hashlib,json,pathlib,time\n"
        "p=pathlib.Path(__file__).parent\nb=b'current'\n(p/'current.mlir').write_bytes(b)\n"
        "(p/'request.json').write_text(json.dumps({\n"
        " 'schema':'merlin.current_scalar_stage_request.v1',\n"
        " 'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}))\ntime.sleep(60)\n"
    )

    def packet(payload, directory):
        return ScalarLeafPatch(
            hashlib.sha256(payload).hexdigest(), "module {}", ({"diagnostic": "typed synthetic transport only"},), {}
        )

    with pytest.raises(ValueError, match="response exceeds"):
        run_command(
            [sys.executable, str(child)],
            directory=tmp_path,
            callback=packet,
            max_source_bytes=1024,
            max_response_bytes=64,
            timeout=5,
        )
    assert not (tmp_path / "response.json").exists()


def test_all_current_members_share_one_table_with_distinct_finishing_factors(tmp_path):
    from merlin.llvmlower.pipeline import apply_passes

    first = source().split("func.func @forward", 1)[1].rsplit("}", 1)[0]
    text = (
        "module {func.func @first"
        + first
        + "func.func @second"
        + first.replace("%factor=arith.constant 3.0", "%factor=arith.constant 5.0")
        + "}"
    )
    current = apply_passes(
        text, "func.func(linalg-fuse-elementwise-ops),func.func(linalg-generalize-named-ops),canonicalize"
    )
    patch = selection().callback(current.encode(), tmp_path)
    assert len(patch.replacements) == 2 and patch.fragment.count('"memref.global"') == 1
    record = json.loads((tmp_path / "binding.json").read_text())
    assert record["family_count"] == 1 and record["physical_table_bytes"] == 6144
    assert {row["quant_factor_bits"] for row in record["members"]} == {word(3.0), word(5.0)}


def test_current_namespace_conflict_and_no_observer_refuse_before_packet(tmp_path):
    from merlin.llvmlower.pipeline import apply_passes

    conflicting = source().rsplit("}", 1)[0] + "func.func private @current_rne()->i1 }"
    current = apply_passes(
        conflicting, "func.func(linalg-fuse-elementwise-ops),func.func(linalg-generalize-named-ops),canonicalize"
    )
    with pytest.raises(ValueError, match="namespace"):
        selection().callback(current.encode(), tmp_path)
    assert not (tmp_path / "binding.json").exists()
    with pytest.raises(ValueError, match="no admitted"):
        selection().callback(b"module {func.func @identity(%x:f32)->f32{return %x:f32}}", tmp_path)


def test_native_narrow_edit_preserves_original_resource_handles_and_trace(tmp_path):
    from merlin.llvmlower.toolchain import m2m_python

    producer_source = (
        source()
        .replace("%x:tensor<7xf32>", "%x:tensor<7xi32>")
        .replace("ins(%x:tensor<7xf32>)", "ins(%x:tensor<7xi32>)")
        .replace("^bb0(%cut:f32", "^bb0(%encoded:i32")
        .replace("%two=arith.constant 2.0:f32", "%cut=arith.sitofp %encoded:i32 to f32\n%two=arith.constant 2.0:f32")
    )
    raw = (
        producer_source.rsplit("}", 1)[0]
        + """memref.global "private" constant @resource_owner : memref<4xi8> = dense_resource<owned_bytes>
    func.func @resource_read()->i8 {
      %m=memref.get_global @resource_owner:memref<4xi8>
      %c=arith.constant 0:index
      %v=memref.load %m[%c]:memref<4xi8>
      return %v:i8
    }}
    {-# dialect_resources: {builtin: {owned_bytes: "0x0100000001030507"}} #-}"""
    )
    raw_path, current_path = tmp_path / "raw.mlir", tmp_path / "current.mlir"
    raw_path.write_text(raw)
    prepare = tmp_path / "prepare.py"
    prepare.write_text(
        "import sys\n"
        "from torch_mlir import ir\n"
        "from torch_mlir.passmanager import PassManager\n"
        "with ir.Context() as ctx:\n"
        " m=ir.Module.parse(open(sys.argv[1]).read())\n"
        " PassManager.parse('builtin.module(func.func(linalg-fuse-elementwise-ops),"
        "func.func(linalg-generalize-named-ops),canonicalize)',ctx).run(m.operation)\n"
        " open(sys.argv[2],'w').write(m.operation.get_asm(print_generic_op_form=True))\n"
    )
    subprocess.run([str(m2m_python()), str(prepare), str(raw_path), str(current_path)], check=True, capture_output=True)
    patch = selection().callback(current_path.read_bytes(), tmp_path)
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(patch.claim()))
    apply = tmp_path / "apply.py"
    apply.write_text(
        "import sys,json,hashlib\nfrom torch_mlir import ir\n_run_stages=lambda *args:None\n"
        + NATIVE
        + """
with ir.Context() as ctx:
 m=ir.Module.parse(open(sys.argv[1]).read())
 original=m.operation
 resource=next(o.operation for o in m.body.operations if o.operation.name=='memref.global')
 attr=resource.attributes['initial_value']
 resource_text=str(resource)
 forward=next(o.operation for o in m.body.operations
              if o.operation.name=='func.func' and str(o.attributes['sym_name'])=='"forward"')
 block=forward.regions[0].blocks[0]
 args=tuple(block.arguments)
 generic=next(o.operation for o in block.operations if o.operation.name=='linalg.generic')
 scalar=generic.regions[0].blocks[0]
 producer=next(o.operation for o in scalar.operations
               if o.operation.name=='arith.sitofp' and str(o.operation.operands[0].type)=='i32')
 producer_text=str(producer)
 before=original.get_asm(print_generic_op_form=True)
 digest=hashlib.sha256(before.encode()).hexdigest()
 packet=json.load(open(sys.argv[2]))
 invalid=dict(packet)
 invalid['replacements']=[*packet['replacements'],*packet['replacements']]
 try:
  _cs_apply(ctx,m,invalid,digest)
 except ValueError:
  pass
 else:
  raise AssertionError('duplicate member admitted')
 assert original.get_asm(print_generic_op_form=True)==before
 _cs_apply(ctx,m,packet,digest)
 assert m.operation==original and resource.parent==original
 assert str(resource)==resource_text and resource.attributes['initial_value']==attr
 assert forward.parent==original and tuple(block.arguments)==args
 assert generic.parent==forward and producer.block==scalar and str(producer)==producer_text
 text=original.get_asm(print_generic_op_form=True,enable_debug_info=True)
 assert 'owned_bytes' in text and 'division-join' in text and 'activation-join' in text
 assert 'prov.scalar_carrier_source_trace' in text and 'fused[' in text
 unchanged=original.get_asm(print_generic_op_form=True)
 try:
  _cs_apply(ctx,m,packet,digest)
 except ValueError:
  pass
 else:
  raise AssertionError('stale one-shot source packet reused')
 assert original.get_asm(print_generic_op_form=True)==unchanged
 open(sys.argv[3],'w').write(text)
 print('PASS ORIGINAL_RESOURCE_PRODUCER_HANDLES_AND_SOURCE_TRACE')
"""
    )
    result = subprocess.run(
        [str(m2m_python()), str(apply), str(current_path), str(packet_path), str(tmp_path / "edited.mlir")],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "PASS ORIGINAL_RESOURCE_PRODUCER_HANDLES_AND_SOURCE_TRACE"


@pytest.mark.parametrize("length", [7, 129])
def test_unselected_default_llvm_bytes_equal_isolated_path(tmp_path, length):
    from merlin.llvmlower.lower import lower_model

    path = tmp_path / "source.mlir"
    path.write_text(source(length))
    before_script = tmp_path / "baseline.py"
    before_script.write_text(
        "import sys\n"
        "from merlin.llvmlower.lower import lower_model_file\n"
        "r=lower_model_file(sys.argv[1],sys.argv[2],targets=())\n"
        "open(sys.argv[3],'wb').write(r.ll_path.read_bytes())\n"
    )
    baseline = tmp_path / "baseline.ll"
    isolated_python = os.environ.get("MERLIN_TEST_SCALAR_CARRIER_PARENT_PYTHON", sys.executable)
    subprocess.run(
        [isolated_python, "-I", "-B", str(before_script), str(path), str(tmp_path / "isolated_path"), str(baseline)],
        check=True,
        capture_output=True,
    )
    after = lower_model(source(length), tmp_path / "overlay_unselected", targets=())
    assert baseline.read_bytes() == after.ll_path.read_bytes()
    assert "source_scalar_carrier" not in after.stats
    assert not (after.workdir / "lower/source_scalar_carrier").exists()


@pytest.mark.parametrize("length,file_api", [(7, False), (31, True), (129, False)])
def test_normal_current_source_native_tail_and_runtime_four_modes(tmp_path, length, file_api):
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.lower import lower_model, lower_model_file
    from merlin.llvmlower.toolchain import clang

    normal = lower_model(source(length), tmp_path / "normal", targets=())
    options = dict(targets=(), source_scalar_carrier=selection())
    if file_api:
        path = tmp_path / "source.mlir"
        path.write_text(source(length))
        selected = lower_model_file(path, tmp_path / "selected", **options)
    else:
        selected = lower_model(source(length), tmp_path / "selected", **options)
    record = selected.stats["source_scalar_carrier"]
    assert record["status"] == "CURRENT_TYPED_EDITS_NATIVE_VERIFIED"
    assert record["physical_table_bytes"] == 6144 and len(record["members"]) == 1
    assert record["predicate_placement"] == "per_point_no_cross_FRM_hoist"
    provider = tmp_path / "provider.c"
    provider.write_text(PROVIDER)
    models = []
    for result in (normal, selected):
        lib = result.workdir / "model.so"
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
                str(lib),
            ],
            check=True,
            capture_output=True,
        )
        models.append(HostModel.load(str(lib)))
    values = np.linspace(1.125, 1.875, length, dtype=np.float32)
    upstream = np.linspace(-19.127, 17.129, length, dtype=np.float32)
    original = values.copy(), upstream.copy()
    saved = HOST.fegetround()
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
            # These independent points are numerically equal; the policy's
            # general guarantee remains approximate with a whole output gate.
            np.testing.assert_array_equal(*outputs)
    finally:
        assert HOST.fesetround(saved) == 0
    np.testing.assert_array_equal(values, original[0])
    np.testing.assert_array_equal(upstream, original[1])
