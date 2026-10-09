"""Independent normal execution has no selected compiler-provider fallback."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import campaign as PC
from merlin_experiments.phase2.component_execution import (
    ComponentIndependentExecution,
    prepare_component_independent_execution,
)
from merlin_experiments.phase2.contracts import StageGateError
from test_edit_authority import package as package


def test_config_fields_cannot_issue_normal_execution_authority():
    from dataclasses import fields

    values = {row.name: None for row in fields(ComponentIndependentExecution)}
    binding = ComponentIndependentExecution(**values)
    with pytest.raises(StageGateError, match="issued independent normal service"):
        binding.verify()


def test_reference_normal_and_analytical_factories_are_removed():
    from merlin_experiments.phase2 import component_analytical, component_execution

    for name in ("ComponentNormalExecutionBinding", "prepare_component_normal_execution",
                 "_prepare_historical_component_normal_execution", "bind_normal_component_execution"):
        assert not hasattr(component_execution, name)
    for name in ("build_selected_component_analytical_provider",
                 "_build_historical_selected_component_analytical_provider"):
        assert not hasattr(component_analytical, name)


def test_independent_normal_route_requires_actual_fresh_authorities():
    with pytest.raises(StageGateError, match="exact fresh baseline"):
        prepare_component_independent_execution(
            baseline_admission=None, independent_runtime=None, view=None, runtime=None,
            contract_root=None, scope=None, output=None,
        )


def test_independent_normal_route_rejects_unowned_candidate_paths(tmp_path):
    from merlin_experiments.phase2.component_execution import _candidate_authority

    with pytest.raises(StageGateError, match="controller-owned candidate edit authority"):
        _candidate_authority(SimpleNamespace(), tmp_path / "arbitrary", SimpleNamespace(configured=True))


def test_independent_execution_reopens_qualified_seed_and_real_edit_contract(package):
    from merlin_experiments.phase2.component_execution import _candidate_authority

    authority, candidate, contract, _output = package
    authority.freeze(candidate, contract, has_iterations=False)
    verified = []

    def reopen_seed(*, candidate):
        verified.append(candidate)
        assert candidate == authority.seed

    admission = SimpleNamespace(qualification=SimpleNamespace(verify=reopen_seed))
    frozen = _candidate_authority(admission, candidate, authority)
    source = candidate / "compiler.py"
    source.write_text(source.read_text().replace("return 1", "return 3"))
    assert _candidate_authority(admission, candidate, authority) == frozen
    assert verified == [authority.seed, authority.seed]
    source.write_text(source.read_text().replace("return 2", "return 9"))
    with pytest.raises(ValueError, match="exceeds host-frozen authority"):
        _candidate_authority(admission, candidate, authority)


def test_independent_owner_has_no_compiler_backend_imports():
    import ast
    import inspect

    from merlin_experiments.phase2 import component_analytical, component_execution, component_providers

    for module in (component_execution, component_analytical, component_providers):
        tree = ast.parse(inspect.getsource(module))
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        assert not any(name and (name.startswith("merlin.backends") or "backend_adapters" in name
                                or "native_model_execution" in name) for name in imports)
        references = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        assert not {"get_backend", "_build_service_for", "prepare_component_feature_provider"} & references


@pytest.mark.skipif(
    not hasattr(PC.oot_runner, "scoped_package_executor"), reason="requires connected core scoped executor"
)
def test_normal_box_preserves_actual_strict_scoped_owner_and_invocation_directory(tmp_path):
    calls = []

    class Executor:
        def build_package(self, pkg, *, timeout):
            calls.append(("build", pkg, timeout))

        def run_entrypoint(self, pkg, name, input_mlir, output_json, **kwargs):
            calls.append(("run", pkg, name, input_mlir, output_json, kwargs))
            return "strict owner result"

    package = SimpleNamespace(directory=tmp_path)
    invocation_root = tmp_path / "private-invocation"
    with PC.oot_runner.scoped_package_executor(Executor()), PC.boxed_entrypoints(None):
        PC.oot_runner.build_package(package, timeout=7)
        result = PC.oot_runner.run_entrypoint(
            package, "parse", Path("input"), timeout=9, invocation_directory=invocation_root
        )
    assert result == "strict owner result"
    assert calls[0] == ("build", package, 7)
    assert calls[1][-1] == {
        "timeout": 9,
        "write_bytecode": False,
        "artifact_profile": None,
        "invocation_directory": invocation_root,
    }
    assert PC.oot_runner.active_package_executor() is None
