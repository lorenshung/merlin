"""Construct feedback from independently qualified support and the fresh seed."""
from pathlib import Path

from .component_baseline import ComponentBaselineAdmission
from .component_runtime import require_independent_runtime
from .contracts import StageGateError


def _inputs(independent_runtime, baseline_admission, role):
    if type(baseline_admission) is not ComponentBaselineAdmission:
        raise StageGateError("independent feedback requires the fresh Phase 1 baseline admission")
    baseline_admission.verify()
    runtime = require_independent_runtime(independent_runtime, required_roles=(role,),
                                          target_descriptor=baseline_admission.target_descriptor)
    verify_binding = getattr(runtime.qualification, "verify_component_binding", None)
    if not callable(verify_binding):
        raise StageGateError("independent feedback requires exact compiler/cohort measurement qualification")
    verify_binding(baseline_admission)
    return runtime


def build_independent_component_analytical_provider(*, independent_runtime, baseline_admission, dependencies, **kwargs):
    """Use the evaluated independent observer; no backend/plugin factory lookup."""
    from .component_analytical import build_component_analytical_provider

    runtime = _inputs(independent_runtime, baseline_admission, "feature_provider")
    for key in ("feature_provider", "baseline", "corpus", "target_descriptor"):
        if key in kwargs:
            raise StageGateError("independent component provider inputs are fixed by fresh baseline admission")
    pins = {Path(path): digest for path, digest in runtime.source_pins}
    for path, digest in dependencies.items():
        path = Path(path)
        if path in pins and pins[path] != digest:
            raise StageGateError("independent component dependency conflicts with evaluated runtime source")
        pins[path] = digest
    return build_component_analytical_provider(
        baseline=baseline_admission.baseline, corpus=baseline_admission.corpus,
        target_descriptor=baseline_admission.target_descriptor, feature_provider=runtime.services.feature_provider,
        baseline_admission=baseline_admission, independent_runtime=runtime, dependencies=pins, **kwargs,
    )


def build_independent_component_cca_provider(*, independent_runtime, baseline_admission):
    """CCA compares the fresh seed with its candidate on identical components."""
    from .component_cca import ComponentCCAProvider
    from .component_workflow import _callable_source

    runtime = _inputs(independent_runtime, baseline_admission, "cca_provider")
    callback = runtime.services.cca_provider
    owner, digest = _callable_source(callback)
    provider = ComponentCCAProvider(
        callback, owner, digest, baseline_admission.baseline, baseline_admission.baseline_sha256,
        baseline_admission, runtime,
    )
    provider.validate()
    return provider
