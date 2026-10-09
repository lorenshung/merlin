"""Actual native invocation observation; no target or compiler qualification."""
import shutil
from types import SimpleNamespace

import pytest
from component_baseline_fixture import synthetic_component_controller as synthetic_component_controller
from merlin_experiments.phase2 import component_observer as O
from merlin_experiments.phase2.contracts import StageGateError, document_sha256
from test_component_workflow import context

from merlin.common import invocation_record
from merlin.perf.component_cost import COMPLETE_STAGES, ComponentCostScope

pytestmark = pytest.mark.usefixtures("synthetic_component_controller")


def actual_records(tmp_path):
    from pathlib import Path

    selected = context(tmp_path)
    source = tmp_path / "independent-native.c"
    source.write_text("int main(void) { return 0; }\n")
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("actual native C compiler unavailable")
    compiler = Path(compiler).resolve()
    root = tmp_path / "controller-private-evidence"
    root.mkdir()
    elf = root / "program.elf"
    result = invocation_record.run(
        [str(compiler), str(source), "-o", str(elf)], directory=root,
        stage="native_object_link", inputs=(source,), outputs=(elf,), capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    first = tuple((root / "invocations").glob("*/invocation.json"))
    result = invocation_record.run(
        [str(elf)], directory=root, stage="native_diagnostic_execution", inputs=(elf,), capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    last = tuple(path for path in (root / "invocations").glob("*/invocation.json") if path not in first)
    return selected, source, root, first + last


def observe(selected, root, paths, **kwargs):
    return O.observe_independent_component_artifacts(
        baseline_admission=selected["baseline_admission"], compiler=selected["candidate"],
        member=selected["component_corpus"].capsules[0], corpus=selected["component_corpus"],
        scope=ComponentCostScope(*(document_sha256(value) for value in ("timer", "accuracy", "inputs"))),
        evidence_root=root, invocation_paths=paths, **kwargs,
    )


def test_actual_native_artifacts_join_without_decoder_or_false_cycle_authority(tmp_path):
    selected, source, root, paths = actual_records(tmp_path)
    observation = observe(selected, root, paths)
    public = observation.public_report()
    assert public["observed_stages"] == ["native_object_link", "native_diagnostic_execution"]
    assert public["input_product_joins"] == 1
    assert public["executable_artifacts"] == 1 and public["static_executable_bytes"] > 0
    assert public["instruction_roles"]["status"] == public["execution_correctness"]["status"] == "UNKNOWN"
    assert public["source_correspondence"]["status"] == "UNKNOWN"
    assert public["cold"] == public["warm"] == {"status": "UNKNOWN", "missing": list(COMPLETE_STAGES)}
    assert public["promotion"] == "DIAGNOSTIC_ONLY"
    assert str(root) not in str(public) and str(source) not in str(public)
    source.write_text("int main(void) { return 1; }\n")
    with pytest.raises(StageGateError, match="changed"):
        observation.verify()


def test_plain_hardware_claims_cannot_grant_independent_instruction_semantics(tmp_path):
    selected, _source, root, paths = actual_records(tmp_path)
    with pytest.raises(StageGateError, match="RTL|independently issued"):
        observe(selected, root, paths, hardware_intake=SimpleNamespace(status="qualified", target="fixture"))


def test_private_invocation_records_cannot_be_read_from_candidate_output(tmp_path):
    selected, _source, root, paths = actual_records(tmp_path)
    with pytest.raises(StageGateError, match="overlaps compiler"):
        observe(selected, selected["candidate"], paths)
    linked = root / "record-link.json"
    linked.symlink_to(paths[0])
    with pytest.raises(StageGateError, match="private controller owner"):
        observe(selected, root, (linked,))
