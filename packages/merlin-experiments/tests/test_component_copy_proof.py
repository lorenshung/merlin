"""Live original ABI selection and real ordinary static copy product controls.

The transport fixture is evaluator-private diagnostic IR, not a fresh compiler
or ISA/runtime authority. Only its provenance admission facet is isolated. No
complete compiler can qualify: original resource/numerical roles stay UNKNOWN.
"""

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase0 import component_compile_sources as S
from merlin_experiments.phase1 import component_compile_roles as R
from merlin_experiments.phase1.component_copy_proof import ComponentCopyProof
from merlin_experiments.phase1.component_pointer_storage import (
    SCHEMA,
    issue_independent_pointer_storage,
    verify_pointer_storage,
)
from merlin_experiments.phase2.component_experiment import ComponentView
from merlin_experiments.phase2.contracts import StageGateError
from test_component_compile_role_transport import transport  # noqa: F401 -- actual stock transport fixture
from test_component_compile_sources import (  # noqa: F401 -- actual native original-source authorities
    configured,
    member,
    plan,
    selected,
    selection,
)

from merlin.common.digest import sha256_file
from merlin.targetgen.contract.pointer_storage import PointerStoragePolicy


@pytest.fixture
def pointer(configured, tmp_path):  # noqa: F811
    plan(configured, [member(), member("private-transfer", "withheld_transfer")])
    roster = S.issue_independent_compile_only_roster(**configured)
    compiler, config = shutil.which("g++"), os.environ.get("MERLIN_TEST_LLVM_CONFIG") or shutil.which("llvm-config")
    if not compiler or not config:
        pytest.skip("requires explicit public native LLVM tools")
    protected = tmp_path / "protected-pointer"
    protected.mkdir()
    path = protected / "policy.json"
    policy = PointerStoragePolicy(
        "row_major_contiguous", "inputs_then_outputs", "disjoint", "static_original_tensor_type", "little", 8
    )
    path.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "hardware_intake_sha256": roster.hardware.sha256,
                "software_intake_sha256": roster.software.sha256,
                "policy": policy.record(),
            }
        )
    )
    return issue_independent_pointer_storage(
        hardware=roster.hardware,
        software=roster.software,
        roster=roster,
        selection=path,
        native_compiler=Path(compiler).resolve(strict=True),
        llvm_config=Path(config).resolve(strict=True),
        forbidden_roots=configured["forbidden_roots"],
        output_root=tmp_path / "pointer-selection",
    )


def test_actual_predeclared_policy_reopens_source_extents_and_refuses_flags_forgery_and_mutation(pointer):
    pointer.verify()
    assert [slot.shape for slot in pointer.bind_member(pointer.roster.members[0]).slots] == [(2, 3), (2, 3)]
    for claimed in (True, json.loads(pointer.receipt_json), replace(pointer)):
        with pytest.raises(StageGateError):
            verify_pointer_storage(
                claimed, hardware=pointer.hardware, software=pointer.software, roster=pointer.roster, view=None
            )
    assert (
        verify_pointer_storage(
            None, hardware=pointer.hardware, software=pointer.software, roster=pointer.roster, view=None
        )
        is None
    )
    policy_source = Path(pointer.source_pins[0].path)
    policy_source.write_text(policy_source.read_text() + "\n")
    with pytest.raises(ValueError, match="source changed"):
        pointer.verify()


@pytest.mark.parametrize("field", ["policy", "source_pin"])
def test_post_issue_nested_object_rebinding_cannot_change_the_original_selection(pointer, tmp_path, field):
    pointer.verify()
    if field == "policy":
        # This remains a supported software choice. Frozen dataclasses alone
        # do not protect an issuer's original value against deliberate bypass.
        object.__setattr__(pointer.policy, "tensor_alignment", 16)
        pointer.policy.record()
    else:
        pin = pointer.source_pins[0]
        replacement = tmp_path / "resealed-policy.json"
        replacement.write_text(Path(pin.path).read_text() + "\n")
        object.__setattr__(pin, "path", str(replacement.resolve()))
        object.__setattr__(pin, "sha256", sha256_file(replacement))
        pin.verify()
    with pytest.raises(StageGateError, match="live independent preauthor selection"):
        pointer.verify()


def test_exact_original_policy_projection_is_required_before_public_author_grant(pointer, tmp_path):
    root = tmp_path / "public-view"
    (root / "contract").mkdir(parents=True)
    path = root / "contract/pointer_storage.json"
    path.write_bytes(pointer.public_json)
    manifest = {
        "schema": "merlin.component_agent_view.v1",
        "generation_sha256": "a" * 64,
        "library_sha256": "b" * 64,
        "members": [
            {
                "path": "contract/pointer_storage.json",
                "sha256": sha256_file(path),
                "n_bytes": path.stat().st_size,
                "role": "contract",
            }
        ],
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    view = ComponentView(root, sha256_file(manifest_path), "a" * 64, "b" * 64)
    assert (
        verify_pointer_storage(
            pointer, hardware=pointer.hardware, software=pointer.software, roster=pointer.roster, view=view
        )
        == pointer.sha256
    )
    # Resealing a structurally consistent manifest never selects a different
    # original ABI or converts candidate-readable JSON into live authority.
    weakened = json.loads(pointer.public_json)
    weakened["policy"]["slot_aliasing"] = "may_alias"
    path.write_text(json.dumps(weakened))
    manifest["members"][0].update(sha256=sha256_file(path), n_bytes=path.stat().st_size)
    manifest_path.write_text(json.dumps(manifest))
    view = replace(view, manifest_sha256=sha256_file(manifest_path))
    with pytest.raises(StageGateError, match="complete original ABI policy"):
        verify_pointer_storage(
            pointer, hardware=pointer.hardware, software=pointer.software, roster=pointer.roster, view=view
        )


def copy_ir(count=6):
    return f"""module {{ llvm.func @control_entry(%a: !llvm.ptr, %b: !llvm.ptr) {{
 %zero = llvm.mlir.constant(0 : i64) : i64
 %one = llvm.mlir.constant(1 : i64) : i64
 %count = llvm.mlir.constant({count} : i64) : i64
 llvm.br ^loop(%zero : i64)
 ^loop(%i: i64):
 %src = llvm.getelementptr %a[%i] : (!llvm.ptr, i64) -> !llvm.ptr, i8
 %dst = llvm.getelementptr %b[%i] : (!llvm.ptr, i64) -> !llvm.ptr, i8
 %value = llvm.load %src {{alignment = 1 : i64}} : !llvm.ptr -> i8
 llvm.store %value, %dst {{alignment = 1 : i64}} : i8, !llvm.ptr
 %next = llvm.add %i, %one : i64
 %done = llvm.icmp "eq" %next, %count : i64
 llvm.cond_br %done, ^exit, ^loop(%next : i64)
 ^exit:
 llvm.return
 }} }}"""


def ordinary_arguments(transport, pointer):  # noqa: F811 -- reused actual stock transport fixture argument
    """Reuse real stock compiler/linker records; never manufacture origin/ISA."""
    abi = pointer.roster.members[0].original_abi
    target = pointer.hardware.target
    original_service = transport["build_service"]
    gate = transport["instruction_check"].admission_service()
    transport["build_service"] = replace(original_service, target=target)
    gate = replace(gate, target=target)
    transport["instruction_check"] = SimpleNamespace(admission_service=lambda: gate)
    transport["roster"] = pointer.roster
    transport["compiler_origin"].inputs.pointer_storage = pointer
    transport["compiler_origin"].inputs.hardware = pointer.hardware
    candidate = transport["candidate"]
    manifest = json.loads((candidate / "manifest.yaml").read_bytes())
    manifest["target"] = target
    (candidate / "manifest.yaml").write_text(json.dumps(manifest))
    slots = (*abi.inputs, *abi.outputs)
    (candidate / "buffer.json").write_text(
        json.dumps(
            {
                "abi_version": "0.1",
                "target": target,
                "commands": [],
                "tensors": {
                    slot.name: {
                        "shape": list(slot.shape),
                        "dtype": slot.dtype,
                        "role": "input" if i < len(abi.inputs) else "output",
                    }
                    for i, slot in enumerate(slots)
                },
                "kernel_abi": {
                    "kind": "whole_program",
                    "outputs": [slot.name for slot in abi.outputs],
                    "args": [
                        {"tensor": slot.name, "access": "read" if i < len(abi.inputs) else "write"}
                        for i, slot in enumerate(slots)
                    ],
                },
            }
        )
    )
    (candidate / "emitted.mlir").write_text(copy_ir())
    return transport


def test_real_source_translation_object_link_static_join_proves_only_copy_facets(pointer, transport):  # noqa: F811
    arguments = ordinary_arguments(transport, pointer)
    evaluation = R.evaluate_component_compile_roles(**arguments)
    observed = evaluation.verify()
    assert observed["compilation_denominator"] == {"required": 2, "linked_and_policy_accepted": 2}
    assert observed["static_denominator"] == {"required": 10, "proved": 6, "unknown": 4}
    assert len(evaluation.static_proofs) == 2 and len(observed["unresolved"]) == 4
    for row in observed["members"]:
        assert row["static_obligations"]["index_bounds"] == "PROVED"
        assert row["static_obligations"]["complete_output_coverage"] == "PROVED"
        assert row["static_obligations"]["resource_legality"] == "UNKNOWN"
        assert row["static_obligations"]["input_numeric_domain"] == "UNKNOWN"
    with pytest.raises(StageGateError, match="remain incomplete"):
        evaluation.require_complete()
    forged = replace(evaluation.static_proofs[0])
    assert type(forged) is ComponentCopyProof
    with pytest.raises(StageGateError, match="actual fixed checker invocation"):
        forged.verify()
    proof = evaluation.static_proofs[0]
    source = proof.transport_report.parent / "build/kernel.llvm.mlir"
    source.write_text(source.read_text().replace("constant(6", "constant(5"))
    with pytest.raises(StageGateError, match="actual product/invocation evidence changed"):
        evaluation.verify()


def test_actual_wrong_stop_bound_discharges_no_required_facet(pointer, transport):  # noqa: F811
    arguments = ordinary_arguments(transport, pointer)
    arguments["candidate"].joinpath("emitted.mlir").write_text(copy_ir(5))
    evaluation = R.evaluate_component_compile_roles(**arguments)
    observed = evaluation.verify()
    assert observed["static_denominator"] == {"required": 10, "proved": 0, "unknown": 4, "refuted": 6}
    assert len(observed["unresolved"]) == 10
    assert all(proof.verify()["status"] == "REFUTED" for proof in evaluation.static_proofs)
    with pytest.raises(StageGateError, match="remain incomplete"):
        evaluation.require_complete()
