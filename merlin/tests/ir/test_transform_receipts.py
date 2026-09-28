"""A semantic receipt binds an exact transform to a reproducible, replayed SMT obligation."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from merlin.verify import HAS_XDSL, HAS_Z3

_TRANSLATOR = os.environ.get("MERLIN_VERIFY_TRANSLATOR")
pytestmark = pytest.mark.skipif(
    not (HAS_XDSL and HAS_Z3 and _TRANSLATOR and Path(_TRANSLATOR).is_file()),
    reason="requires xdsl, z3, and an explicit MERLIN_VERIFY_TRANSLATOR",
)


@pytest.fixture(autouse=True)
def explicit_test_target(tmp_path, monkeypatch):
    """Select self-contained reference metadata; installed Merlin has no target default."""
    root = tmp_path / "receipt_target"
    contracts = root / "contracts"
    contracts.mkdir(parents=True)
    (contracts / "target_contract.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "toy_npu",
                "features": ["resident_packed_tensor", "accumulator_commit", "command_buffer", "metrics"],
                "runtime": {"backends": ["simulator"], "default_backend": "simulator"},
                "capabilities": {"resident_storage_bytes": 131072},
            }
        )
    )
    (contracts / "dialect_plan.yaml").write_text(
        yaml.safe_dump(
            {
                "target": "toy_npu",
                "dialect_name": "toynpu",
                "types": [{"name": "resident_tensor"}, {"name": "accumulator"}],
                "construction": {"matmul_rhs_typed": True, "matmul_vl_policy": False},
                "runtime_opcodes": {
                    "res_pack": "RES_PACK",
                    "matmul": "MATMUL_RESIDENT",
                    "commit": "COMMIT",
                    "evict": "EVICT",
                    "vector_map": "VECTOR_MAP",
                    "vector_reduce": "VREDUCE",
                },
                "lowering": [
                    {"from": "interface.resident_pack", "to": "toynpu.res_pack"},
                    {"from": "interface.matmul", "to": "toynpu.matmul"},
                    {"from": "interface.commit", "to": "toynpu.commit"},
                    {"from": "interface.resident_evict", "to": "toynpu.evict"},
                    {"from": "interface.vector_map", "to": "toynpu.vector_map"},
                    {"from": "interface.vector_reduce", "to": "toynpu.vector_reduce"},
                ],
            }
        )
    )
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(root))
    monkeypatch.delenv("MERLIN_TARGET_CONTRACT", raising=False)


def _tool() -> tuple[str, str]:
    assert _TRANSLATOR is not None
    return _TRANSLATOR, hashlib.sha256(Path(_TRANSLATOR).read_bytes()).hexdigest()


def _source_pair():
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerAttr, i32
    from xdsl.dialects.linalg import ops as linalg_ops

    from merlin.xdsl_dialects.lowering.pipeline import build_input_module, lower_module

    source = build_input_module(reuse=2, m=2, k=2, n=2)
    func = next(op for op in source.walk() if op.name == "func.func")
    block = func.body.block
    for matmul in [op for op in block.ops if op.name == "linalg.quantized_matmul"]:
        init = matmul.operands[-1]
        zero = arith.ConstantOp(IntegerAttr(0, i32))
        fill = linalg_ops.FillOp(inputs=(zero.result,), outputs=(init,), res=(matmul.results[0].type,))
        block.insert_op_before(zero, matmul)
        block.insert_op_before(fill, matmul)
        matmul.operands = (*matmul.operands[:-1], fill.results[0])
    result = lower_module(source, target="toy_npu")
    return result.input_module, result.interface_module


def _backend_pair():
    from merlin.verify.evaluate import _finish_lowering, _lower_to_interface

    interface, contract = _lower_to_interface(2, 2, 2, 2)
    return interface, _finish_lowering(interface, contract)


@pytest.mark.parametrize(
    ("boundary", "pair"),
    [
        ("linalg_to_interface", _source_pair),
        ("interface_to_command_buffer", _backend_pair),
    ],
)
def test_real_transform_receipt_proves_and_replays_exact_artifacts(boundary, pair):
    from merlin.verify.receipts import TransformReceipt, qualify_receipt, verify_transformation

    source, target = pair()
    translator, digest = _tool()
    receipt = verify_transformation(boundary, source, target, translator=translator, expected_translator_sha256=digest)
    assert receipt.verified, (receipt.status, receipt.reason)
    assert receipt.smt2_sha256
    assert receipt.source["sha256"] != receipt.target["sha256"]
    assert receipt.toolchain["mlir_translate_sha256"] == digest
    assert receipt.toolchain["z3_version"]
    assert receipt.toolchain["verifier_sha256"]
    assert receipt.typed_signatures["source"]["function_count"] == 1
    assert receipt.typed_signatures["source"]["inputs"]
    assert qualify_receipt(receipt, source, target, translator=translator)
    loaded = TransformReceipt.from_dict(json.loads(json.dumps(receipt.to_dict())))
    assert qualify_receipt(loaded, source, target, translator=translator)
    assert not qualify_receipt(replace(receipt, method="syntax_verification"), source, target, translator=translator)


def test_source_mutation_invalidates_an_old_receipt_and_yields_counterexample():
    from xdsl.dialects.builtin import IntegerAttr, i32

    from merlin.verify.receipts import qualify_receipt, verify_transformation

    source, target = _source_pair()
    translator, digest = _tool()
    old = verify_transformation(
        "linalg_to_interface", source, target, translator=translator, expected_translator_sha256=digest
    )
    assert old.verified
    zero_point = next(op for op in source.walk() if op.name == "arith.constant")
    zero_point.properties["value"] = IntegerAttr(7, i32)
    assert not qualify_receipt(old, source, target, translator=translator)
    current = verify_transformation(
        "linalg_to_interface", source, target, translator=translator, expected_translator_sha256=digest
    )
    assert current.status == "refuted", (current.status, current.reason)
    assert current.counterexample and current.counterexample["inputs"]
    assert current.source["sha256"] != old.source["sha256"]


def test_bare_tensor_empty_never_produces_a_qualified_receipt():
    from merlin.verify.receipts import qualify_receipt, verify_transformation
    from merlin.xdsl_dialects.lowering.pipeline import lower_repeated_rhs_matmul

    legacy = lower_repeated_rhs_matmul(reuse=2, m=2, k=2, n=2, target="toy_npu")
    translator, digest = _tool()
    receipt = verify_transformation(
        "linalg_to_interface",
        legacy.input_module,
        legacy.interface_module,
        translator=translator,
        expected_translator_sha256=digest,
    )
    assert receipt.status == "unsupported"
    assert "tensor.empty with unspecified contents" in (receipt.reason or "")
    assert not receipt.verified
    assert not qualify_receipt(receipt, legacy.input_module, legacy.interface_module, translator=translator)


def test_returning_wrong_commit_order_abstains():
    from merlin.verify.receipts import verify_transformation

    source, target = _source_pair()
    func = next(op for op in target.walk() if op.name == "func.func")
    return_op = next(op for op in func.body.block.ops if op.name == "func.return")
    return_op.operands = (return_op.operands[1], return_op.operands[0])
    translator, digest = _tool()
    receipt = verify_transformation(
        "linalg_to_interface", source, target, translator=translator, expected_translator_sha256=digest
    )
    assert receipt.status == "unsupported"
    assert "func.return" in (receipt.reason or "")


def test_backend_mutation_refutes_and_changed_digest_cannot_qualify():
    from merlin.verify.receipts import qualify_receipt, verify_transformation

    source, target = _backend_pair()
    translator, digest = _tool()
    old = verify_transformation(
        "interface_to_command_buffer", source, target, translator=translator, expected_translator_sha256=digest
    )
    assert old.verified
    changed = copy.deepcopy(target)
    from merlin.verify.faults import CB_CORPUS

    next(f for f in CB_CORPUS if f.name == "cb_swapped_matmul_operands").mutate(changed)
    current = verify_transformation(
        "interface_to_command_buffer", source, changed, translator=translator, expected_translator_sha256=digest
    )
    assert current.status == "refuted", (current.status, current.reason)
    assert current.counterexample and current.counterexample["inputs"]
    assert not qualify_receipt(old, source, changed, translator=translator)


def test_wrong_tool_pin_and_solver_unknown_never_qualify(monkeypatch):
    from merlin.verify import refine
    from merlin.verify.receipts import qualify_receipt, verify_transformation
    from merlin.verify.smt_export import Verdict

    source, target = _source_pair()
    translator, _ = _tool()
    mismatch = verify_transformation(
        "linalg_to_interface", source, target, translator=translator, expected_translator_sha256="0" * 64
    )
    assert mismatch.status == "unavailable"
    assert not mismatch.verified

    monkeypatch.setattr(refine, "validate_pass", lambda *args, **kwargs: Verdict("unknown", reason="timeout"))
    unknown = verify_transformation("linalg_to_interface", source, target, translator=translator)
    assert unknown.status == "unknown" and unknown.reason == "timeout"
    assert not qualify_receipt(unknown, source, target, translator=translator)


def test_float_source_is_explicitly_unsupported():
    from xdsl.dialects.builtin import FunctionType, ModuleOp, TensorType, f32
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.ir import Block, Region

    from merlin.verify.receipts import verify_transformation

    _, target = _source_pair()
    tensor_type = TensorType(f32, [2, 2])
    block = Block(arg_types=[tensor_type])
    block.add_op(ReturnOp(block.args[0]))
    source = ModuleOp([FuncOp("float_source", FunctionType.from_lists([tensor_type], [tensor_type]), Region([block]))])
    translator, digest = _tool()
    receipt = verify_transformation(
        "linalg_to_interface", source, target, translator=translator, expected_translator_sha256=digest
    )
    assert receipt.status == "unsupported", (receipt.status, receipt.reason)
    assert "f32" in (receipt.reason or "")
