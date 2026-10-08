"""A mixed mesh+host emission must be recognised as executable from its ARTIFACT.

``host_lane_program_emitted`` is written on one lane only -- the pure-host builder. A program whose
regions split across the mesh and the host takes the mixed lane, which emits its artifact and
returns before that key is set. Reading its absence as "not a program" refused a real 109-command
whole-model emission, so the check now derives agreement between the artifact's entry arity and the
``kernel_abi`` the buffer declares.

Each negative case here is a MUTATION of the positive one, so a check that cannot fail is visible.
"""

from __future__ import annotations

import pytest

from merlin.targetgen.bundle_harness import (
    EMITTED_PROGRAM_KEY,
    BundleHarnessError,
    emitted_entry_arity,
    is_executable_emission,
    require_executable_emission,
)


def _artifact(parameters: int) -> str:
    """A printed LLVM-dialect module whose single entry takes ``parameters`` pointers."""
    pointers = ", ".join(["!llvm.ptr"] * parameters)
    block_args = ", ".join(f"%arg{i}: !llvm.ptr" for i in range(parameters))
    return (
        '"builtin.module"() ({\n'
        '  "llvm.func"() <{unnamed_addr = 0 : i64, sym_name = "kernel", '
        f"function_type = !llvm.func<void ({pointers})>, CConv = #llvm.cconv<ccc>, "
        'linkage = #llvm.linkage<"external">, visibility_ = 0 : i64}> ({\n'
        f"  ^bb0({block_args}):\n"
        '    "llvm.return"() : () -> ()\n'
        "  }) : () -> ()\n"
        "}) : () -> ()\n"
    )


def _pretty_artifact(parameters: int, *, name: str = "gemmini_kernel", definition: bool = True) -> str:
    """Registered LLVM pretty syntax, as emitted by a mixed whole-program builder."""
    pointers = ", ".join(f"%arg{i}: !llvm.ptr" for i in range(parameters))
    body = " { llvm.return }" if definition else ""
    return f"module {{ llvm.func @{name}({pointers}){body} }}"


def _mixed_buffer(arguments: int) -> dict:
    """A mixed-lane buffer: real commands, a whole-program ABI, and NO emitted-program key."""
    return {
        "commands": [{"opcode": "CONV2D"}] * 3 + [{"opcode": "COMMIT"}],
        "kernel_abi": {
            "kind": "whole_program",
            "args": [{"tensor": f"t{i}", "access": "read"} for i in range(arguments)],
            "outputs": ["t0"],
        },
        "params": {"mesh_regions": [{}], "host_lane_regions": [{}]},
    }


def test_entry_arity_is_parsed_from_the_printed_signature() -> None:
    assert emitted_entry_arity(_artifact(393)) == 393
    assert emitted_entry_arity(_artifact(1)) == 1


def test_arity_ignores_nesting_inside_a_parameter_type() -> None:
    """A struct-typed parameter carries commas of its own; they are not parameter separators."""
    text = "module { llvm.func @kernel(%a: !llvm.struct<(i32, i32, i64)>, %b: !llvm.ptr) { llvm.return } }"
    assert emitted_entry_arity(text) == 2


def test_artifact_with_no_function_definition_has_no_arity() -> None:
    assert emitted_entry_arity('"builtin.module"() ({}) : () -> ()') is None


def test_mixed_lane_emission_is_executable_when_the_artifact_agrees() -> None:
    buffer = _mixed_buffer(393)
    assert EMITTED_PROGRAM_KEY not in buffer["params"]
    ok, why_not = is_executable_emission(buffer, artifact_text=_artifact(393))
    assert ok, why_not
    require_executable_emission(buffer, artifact_text=_artifact(393))


def test_pretty_llvm_definition_is_executable_when_its_abi_agrees() -> None:
    # Production emits this registered spelling; the older generic-only scan missed it.
    ok, why_not = is_executable_emission(_mixed_buffer(2), artifact_text=_pretty_artifact(2))
    assert ok, why_not


def test_only_the_named_defined_entry_can_satisfy_a_mixed_emission() -> None:
    artifact = (
        "module { "
        "llvm.func @helper(%arg0: !llvm.ptr) { llvm.return } "
        "llvm.func @candidate_entry(%arg0: !llvm.ptr) { llvm.return } "
        "}"
    )
    buffer = _mixed_buffer(1)
    assert is_executable_emission(buffer, artifact_text=artifact, entry_symbol="candidate_entry")[0]
    assert not is_executable_emission(buffer, artifact_text=artifact, entry_symbol="missing")[0]
    assert not is_executable_emission(buffer, artifact_text=artifact)[0], "ambiguous helper must not be an entry"
    declaration = _pretty_artifact(1, name="candidate_entry", definition=False)
    assert not is_executable_emission(buffer, artifact_text=declaration, entry_symbol="candidate_entry")[0]
    spoof = "module {} // llvm.func @candidate_entry(%arg0: !llvm.ptr) { llvm.return }"
    assert not is_executable_emission(buffer, artifact_text=spoof, entry_symbol="candidate_entry")[0]
    duplicate = (
        "module { llvm.func @candidate_entry(%arg0: !llvm.ptr) "
        "llvm.func @candidate_entry(%arg0: !llvm.ptr) { llvm.return } }"
    )
    assert not is_executable_emission(buffer, artifact_text=duplicate, entry_symbol="candidate_entry")[0]
    assert not is_executable_emission(buffer, artifact_text="module { llvm.func @candidate_entry(")[0]
    assert not is_executable_emission(buffer, artifact_text=_pretty_artifact(1), entry_symbol=1)[0]
    with pytest.raises(BundleHarnessError):
        require_executable_emission(buffer, artifact_text=artifact, entry_symbol="missing")


def test_same_buffer_without_its_artifact_is_still_refused() -> None:
    """The buffer alone is byte-identical whether or not an artifact was built beside it."""
    ok, why_not = is_executable_emission(_mixed_buffer(393))
    assert not ok
    assert EMITTED_PROGRAM_KEY in why_not
    assert "artifact_text=" in why_not


def test_arity_disagreement_is_refused_and_says_both_numbers() -> None:
    ok, why_not = is_executable_emission(_mixed_buffer(393), artifact_text=_artifact(392))
    assert not ok
    assert "392" in why_not and "393" in why_not


def test_artifact_without_an_entry_is_refused() -> None:
    ok, why_not = is_executable_emission(_mixed_buffer(393), artifact_text='"builtin.module"() ({}) : () -> ()')
    assert not ok
    assert "no entry function" in why_not


def test_a_buffer_declaring_no_abi_cannot_be_checked_by_an_artifact() -> None:
    """An artifact must be checked AGAINST something; an empty ABI proves nothing."""
    buffer = _mixed_buffer(393)
    buffer["kernel_abi"] = {"kind": "whole_program", "args": [], "outputs": []}
    ok, why_not = is_executable_emission(buffer, artifact_text=_artifact(393))
    assert not ok
    assert "cannot be shown to agree" in why_not


def test_a_declined_buffer_is_refused_even_with_an_artifact() -> None:
    """A decline outranks every other evidence: the compiler said it could not build this."""
    buffer = _mixed_buffer(393)
    buffer["declined"] = {"reason": "no scalar format for f8", "op": "linalg.generic"}
    ok, why_not = is_executable_emission(buffer, artifact_text=_artifact(393))
    assert not ok
    assert "DECLINED" in why_not
    with pytest.raises(BundleHarnessError):
        require_executable_emission(buffer, artifact_text=_artifact(393))


def test_pure_host_declaration_still_needs_no_artifact() -> None:
    buffer = _mixed_buffer(393)
    buffer["params"][EMITTED_PROGRAM_KEY] = True
    assert is_executable_emission(buffer)[0]
    buffer["params"][EMITTED_PROGRAM_KEY] = False
    assert not is_executable_emission(buffer, artifact_text=_artifact(393))[0]
