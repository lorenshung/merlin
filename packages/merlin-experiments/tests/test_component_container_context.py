"""Actual selected compiler commands in private runtime diagnostic context.

The source-preparation fixture explicitly disables hardware admission only for
these diagnostic controls. It does not mint runtime, origin or numeric authority.
Private upstream source defects still run ordinary lowering and original gates.
"""

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.component_package_execution import selected_compiler_transport
from merlin_experiments.phase2.component_runtime import inventory_runtime
from merlin_experiments.phase2.component_runtime_qualification import RuntimeControlRefusal
from merlin_experiments.phase2.contracts import StageGateError
from test_component_compile_role_transport import transport as _compile_role_fixture
from test_component_experiment import _view
from test_component_runtime_support import prepared as prepared
from test_container_transport import selected_native as selected_native

from merlin.common import invocation_record


def test_saved_runtime_or_transport_declarations_cannot_select_compiler_commands():
    with pytest.raises(StageGateError, match="independently issued runtime"):
        selected_compiler_transport(SimpleNamespace(qualified=True), view=None, runtime=())


def test_readiness_saved_transport_refuses_before_command():
    from merlin_experiments.phase1.component_tool_readiness import probe_shared_tools

    with pytest.raises(StageGateError, match="actual prepared compiler"):
        probe_shared_tools(SimpleNamespace(), (), compiler_transport={"status": "qualified"})


def test_unused_or_declaration_only_context_grants_refuse(prepared, tmp_path):
    with pytest.raises(StageGateError, match="explicit prepared command"):
        replace(prepared, compiler_view=_view(tmp_path))._compiler_commands()
    with pytest.raises(StageGateError, match="exact prepared transport"):
        replace(prepared, container_transport={"status": "qualified"})._compiler_commands()


@pytest.fixture
def stock_compile_case(tmp_path, monkeypatch):
    separate = tmp_path / "separate-compile-owner"
    separate.mkdir()
    return _compile_role_fixture.__wrapped__(separate, monkeypatch)


@pytest.fixture
def native_context(prepared, tmp_path, selected_native):
    import os

    import immutabledict
    import ordered_set
    import typing_extensions
    import xdsl

    selected_python = os.environ.get("MERLIN_TEST_COMPONENT_PYTHON")
    selected_stdlib = os.environ.get("MERLIN_TEST_COMPONENT_STDLIB")
    if not selected_python or not selected_stdlib:
        pytest.skip("requires explicitly selected public Python interpreter and standard-library roots")
    transport, _compiler = selected_native
    import json

    metadata = invocation_record.run(
        [
            selected_python,
            "-I",
            "-c",
            "import json,sysconfig;print(json.dumps({name:sysconfig.get_path(name) for name in ('stdlib','purelib')}))",
        ],
        directory=tmp_path / "interpreter-layout",
        stage="selected_interpreter_layout",
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    layout = json.loads(metadata.stdout)
    assert Path(layout["stdlib"]).resolve() == Path(selected_stdlib).resolve()
    purelib = layout["purelib"]
    # Explicit public interpreter/stdlib and four upstream xDSL dependencies.
    # No site-package root, editable .pth, checkout, host home or credential tree.
    files = ((Path(typing_extensions.__file__).resolve(), purelib + "/typing_extensions.py"),)
    trees = (
        (Path(selected_stdlib), layout["stdlib"]),
        *(
            (Path(module.__file__).resolve().parent, purelib + "/" + module.__name__)
            for module in (xdsl, immutabledict, ordered_set)
        ),
    )
    runtime = inventory_runtime(
        files=files,
        trees=trees,
        executables=(
            (Path(selected_python), "/usr/bin/python3"),
            (Path("/usr/bin/bwrap"), "/usr/bin/bwrap"),
            *(
                (path, str(Path(layout["stdlib"]) / path.relative_to(selected_stdlib)))
                for path in sorted(Path(selected_stdlib).glob("lib-dynload/*.so"))
            ),
        ),
    )
    view = _view(tmp_path)
    context = replace(prepared, container_transport=transport, compiler_view=view, compiler_runtime=runtime)
    assert context._compiler_commands()["transport_sha256"] == transport.sha256
    return context


@pytest.mark.parametrize("case", ["source_correspondence.negative", "original_output_roster.negative"])
def test_actual_runtime_context_uses_selected_transport_and_original_source_gates(native_context, tmp_path, case):
    context = native_context
    fixture = context.prepare_control(case, tmp_path / "source-control")
    with pytest.raises(RuntimeControlRefusal) as rejected:
        context.grade(**fixture.grade_arguments)
    assert rejected.value.case_id == case
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == sha for path, sha in rejected.value.evidence_files)
    records = [invocation_record.verify(path) for path in fixture.evidence_root.rglob("invocation.json")]
    dispatches = [
        row
        for row in records
        if row["kind"] == "python_call" and row["callable"].endswith("PreparedContainerTransport.execute")
    ]
    assert len(dispatches) >= 3
    assert any(row["stage"] == "container_command" for row in records)
    context.verify_control(fixture)
    import json

    reports = [json.loads(path.read_text()) for path in fixture.evidence_root.rglob("transport.json")]
    assert reports and all(
        row["native_cleanup"] == "COMPLETE" and row["owned_container_removal"] == "OBSERVED" for row in reports
    )


def test_active_unselected_executor_refuses_before_any_compiler(native_context, tmp_path):
    from merlin.targetgen import package_runtime

    context = native_context
    fixture = context.prepare_control("source_correspondence.positive", tmp_path / "source-control")
    with package_runtime.scoped_package_executor(
        SimpleNamespace(
            container_transport=None,
            run_entrypoint=lambda *args, **kwargs: pytest.fail("unselected compiler ran"),
            build_package=lambda *args, **kwargs: pytest.fail("unselected build ran"),
        )
    ):
        with pytest.raises(StageGateError, match="differs from the prepared command"):
            context.grade(**fixture.grade_arguments)
    assert not list(fixture.evidence_root.rglob("transport.json"))


def test_actual_fresh_tool_readiness_uses_selected_container_without_model(native_context, tmp_path):
    from merlin_experiments.phase1 import component_origin as origin
    from merlin_experiments.phase2.component_experiment import strict_tool_policy

    context = native_context
    candidate = tmp_path / "inert-tool-candidate"
    candidate.mkdir()
    output = tmp_path / "readiness-owner"
    output.mkdir()
    command = ("/usr/bin/python3", "-I", "-B", "-c", "print('ACTUAL_PRIVATE_TOOL_READINESS')")
    probe = origin.FreshToolProbe(
        "host_compiler", command, hashlib.sha256(b"ACTUAL_PRIVATE_TOOL_READINESS\n").hexdigest()
    )
    inputs = SimpleNamespace(
        candidate=candidate,
        output=output,
        runtime=context.compiler_runtime,
        compiler_transport=context.container_transport,
        readiness=(probe,),
    )
    sandbox = next(row.source for row in context.compiler_runtime if row.destination == "/usr/bin/bwrap")
    policy = strict_tool_policy(
        context.compiler_view,
        candidate,
        runtime=context.compiler_runtime,
        candidate_destination=str(candidate),
        bwrap_binary=sandbox,
    )
    origin._probe_shared_tools(inputs, policy)
    records = [invocation_record.verify(path) for path in output.rglob("invocation.json")]
    (dispatch,) = [row for row in records if row["stage"] == "fresh_phase1_host_compiler"]
    assert dispatch["kind"] == "python_call" and dispatch["outcome"] == "returned"
    assert any(row["stage"] == "container_command" for row in records)
    assert not list(candidate.iterdir())


def test_actual_compile_role_transport_preserves_unknown_static_facets(native_context, stock_compile_case, monkeypatch):
    from merlin_experiments.phase1 import component_compile_roles as roles
    from merlin_experiments.phase1.component_package_execution import qualified_package_execution

    # Existing fixture isolates origin/ISA admission; the actual ordinary source,
    # translation, object/link and original static denominator all remain real.
    # This cannot qualify a compiler, ISA or hardware runtime.
    inputs = stock_compile_case["compiler_origin"].inputs
    inputs.view = native_context.compiler_view
    inputs.runtime = native_context.compiler_runtime
    inputs.compiler_transport = native_context.container_transport
    monkeypatch.setattr(roles, "qualified_package_execution", qualified_package_execution)
    evaluated = roles.evaluate_component_compile_roles(**stock_compile_case)
    document = evaluated.verify()
    assert document["status"] == "incomplete" and document["unresolved"]
    assert document["numerical_execution"] == "not_attempted" and document["performance"] == "not_attempted"
    records = [invocation_record.verify(path) for path in evaluated.receipt.parent.rglob("invocation.json")]
    assert any(
        row["kind"] == "python_call" and row["callable"].endswith("PreparedContainerTransport.execute")
        for row in records
    )
    assert any(row["stage"] == "container_command" for row in records)
    assert document["compilation_denominator"]["linked_and_policy_accepted"] > 0
