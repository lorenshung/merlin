"""Real canonical schemas drive bounded original-source logical alias cases.

Native tests require explicitly selected public framework/capture sources;
absence is an explicit skip, never replacement observations or schema tables.
"""

import copy
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml
from merlin_experiments.phase0 import component_automatic as A
from merlin_experiments.phase0 import operator_schema_intake as O
from merlin_experiments.phase0.rtl_intake import RtlIntakeRefusal

from merlin.common import invocation_record as I
from merlin.common.paths import module_source_path
from merlin.targetgen.capsule_inputs import materialize_capsule_leaves
from merlin.targetgen.frontend_operator_effects import original_operator_effects
from merlin.targetgen.frontend_trace import _digest

_fixture_path = Path(__file__).with_name("test_component_automatic.py")
_fixture_spec = importlib.util.spec_from_file_location("private_automatic_source_fixtures", _fixture_path)
automatic_fixtures = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(automatic_fixtures)

automatic = automatic_fixtures.automatic
independent = automatic_fixtures.independent
selected = automatic_fixtures.selected
write = automatic_fixtures.write


@pytest.fixture(scope="module")
def native_sources(tmp_path_factory):
    names = ("MERLIN_TEST_TORCH_PYTHON", "MERLIN_TEST_M2M_ROOT", "MERLIN_TEST_OPERATOR_DECLARATIONS")
    if any(not os.environ.get(name) for name in names):
        pytest.skip("native schema controls need explicit protected public Python/capture/declaration sources")
    python, capture_root, declarations = (Path(os.environ[name]).absolute() for name in names)
    destination = tmp_path_factory.mktemp("actual-schema-source")
    script = destination / "capture.py"
    script.write_text("""import sys,json,importlib.util
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import torch
from m2m.capture.trace import snapshot_exported_program
spec=importlib.util.spec_from_file_location("schema_observer",sys.argv[2])
observer=importlib.util.module_from_spec(spec);spec.loader.exec_module(observer)
class Model(torch.nn.Module):
 def forward(self,A,W0,W1):
  return A@W0,A@W1,A.clone(),A.reshape(2,3)
class Mutation(torch.nn.Module):
 def forward(self,A,B):
  return torch.ops.aten.copy_.default(A,B)
out={}
cases=[("alias",Model(),(torch.arange(6,dtype=torch.int8).reshape(2,3),
        torch.ones(3,2,dtype=torch.int8),torch.ones(3,2,dtype=torch.int8))),
       ("mutation",Mutation(),(torch.ones(2,3,dtype=torch.int8),torch.zeros(2,3,dtype=torch.int8)))]
for name,model,inputs in cases:
 graph=snapshot_exported_program(torch.export.export(model,inputs),stage="original")
 operations=sorted({node["target"] for node in graph["nodes"] if node["op"]=="call_function"})
 request={"namespace":"aten","captured_schemas":graph["operator_schemas"],"operations":operations}
 observation=observer.observe(request,declarations=Path(sys.argv[3]).read_bytes())
 out[name]={"trace":{"schema":"m2m.frontend_trace.v1","graphs":{"original":graph}},"observation":observation}
print(json.dumps(out,sort_keys=True))
""")
    result = I.run(
        [
            str(python),
            "-I",
            str(script),
            str(capture_root),
            str(module_source_path("merlin.targetgen.torch_schema_observer")),
            str(declarations),
        ],
        directory=destination,
        stage="actual_original_schema_controls",
        inputs=(script, declarations, module_source_path("merlin.targetgen.torch_schema_observer")),
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
        capture_output=True,
        timeout=60,
    )
    result.check_returncode()
    return json.loads(result.stdout), python, declarations


def resign(trace):
    graph = trace["graphs"]["original"]
    graph["sha256"] = _digest({key: value for key, value in graph.items() if key != "sha256"})


def test_actual_capture_canonical_schema_and_registered_runtime_bind_alias(native_sources):
    documents, _, _ = native_sources
    source = documents["alias"]
    observed = original_operator_effects(source["trace"], source["observation"])
    assert observed.effect_classes == ("may_alias_result",)
    assert observed.unknowns() == []
    (witness,) = observed.witnesses()
    graph = source["trace"]["graphs"]["original"]
    call = next(node for node in graph["nodes"] if node["id"] == witness["node"])
    assert witness["input_value"] == call["args"][0]["value_id"]
    assert witness["result_value"] == call["results"][0]["id"]
    assert witness["argument_path"] == "args/0"
    assert set(observed.public_semantics()) == {"graph_sha256", "effect_classes"}
    assert "physical" not in observed.effect_classes


def test_actual_source_mutation_is_mandatory_information_without_an_arithmetic_permission(native_sources):
    documents, _, _ = native_sources
    source = documents["mutation"]
    observed = original_operator_effects(source["trace"], source["observation"])
    assert "may_write_argument" in observed.effect_classes
    assert any(row["kind"] == "may_write_argument" and row["argument_path"] == "args/0" for row in observed.witnesses())


def test_actual_native_observer_refuses_a_mismatched_captured_schema(native_sources, tmp_path):
    documents, python, declarations = native_sources
    graph = documents["alias"]["trace"]["graphs"]["original"]
    schemas = dict(graph["operator_schemas"])
    schemas["aten.reshape.default"] = schemas["aten.clone.default"]
    request = write(
        tmp_path / "mismatched-capture.json",
        {"namespace": "aten", "captured_schemas": schemas, "operations": sorted(schemas)},
    )
    observer = module_source_path("merlin.targetgen.torch_schema_observer")
    result = I.run(
        [str(python), "-I", str(observer), str(request), str(declarations)],
        directory=tmp_path,
        stage="mismatched_original_schema_control",
        inputs=(observer, request, declarations),
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
        capture_output=True,
        timeout=60,
    )
    result.check_returncode()
    row = next(row for row in json.loads(result.stdout)["rows"] if row["target"] == "aten.reshape.default")
    assert row["status"] == "unknown"
    assert "canonical public source declaration" in row["reason"]


@pytest.mark.parametrize(
    "defect",
    ["captured_schema", "missing_schema", "required_argument", "tuple_return", "wildcard_alias", "alias_transition"],
)
def test_original_or_schema_defects_cannot_mint_alias_relations(native_sources, defect):
    documents, _, _ = native_sources
    source = copy.deepcopy(documents["alias"])
    graph = source["trace"]["graphs"]["original"]
    call = next(node for node in graph["nodes"] if node["target"] == "aten.reshape.default")
    row = next(row for row in source["observation"]["rows"] if row["target"] == call["target"])
    if defect == "captured_schema":
        graph["operator_schemas"][call["target"]] = "different"
    elif defect == "missing_schema":
        graph["operator_schemas"].pop(call["target"])
    elif defect == "required_argument":
        call["args"].pop()
    elif defect == "tuple_return":
        row["returns"][0]["type"] = "List[Tensor]"
    elif defect == "wildcard_alias":
        row["arguments"][0]["alias"]["before"] = ["*"]
        row["arguments"][0]["alias"]["after"] = ["*"]
    else:
        row["arguments"][0]["alias"]["after"] = ["other"]
    resign(source["trace"])
    observed = original_operator_effects(source["trace"], source["observation"])
    assert observed.effect_classes == ()
    assert observed.unknowns()


@pytest.fixture
def effect_generation(native_sources, monkeypatch, request, tmp_path):
    documents, python, declarations = native_sources
    monkeypatch.setattr(automatic_fixtures, "example", lambda **kwargs: copy.deepcopy(documents["alias"]["trace"]))
    options = request.getfixturevalue("automatic")
    # Exact tracked fixture selection uses actual observed public bytes. It
    # does not attest the configured URL or historical installed-library build.
    checkout = tmp_path / "public-declarations"
    checkout.mkdir()
    source = checkout / "native_functions.yaml"
    source.write_bytes(declarations.read_bytes())
    for args in (
        ("init",),
        ("config", "user.name", "Independent fixture"),
        ("config", "user.email", "fixture@example.invalid"),
        ("remote", "add", "origin", "https://example.invalid/public-schema-fixture"),
        ("add", "native_functions.yaml"),
        ("commit", "-m", "Pin public declaration bytes"),
    ):
        subprocess.run(["git", "-C", str(checkout), *args], check=True, capture_output=True)
    commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"]).decode().strip()
    selection = write(
        tmp_path / "schema-selection.json",
        {
            "schema": O.SELECTION_SCHEMA,
            "status": "reviewed",
            "software_intake_sha256": options["software_intake"].sha256,
            "namespace": "aten",
            "python": str(python),
            "canonical_source": {"checkout": str(checkout), "commit": commit, "path": str(source)},
        },
    )
    forbidden = tmp_path / "issued-forbidden"
    forbidden.mkdir()
    intake = O.issue_independent_operator_schema_intake(
        software=options["software_intake"],
        selection=selection,
        forbidden_roots=(forbidden,),
        output=tmp_path / "issued-schemas",
    )
    policy = yaml.safe_load(options["component_coverage"].read_bytes())
    policy.update(schema=A.EFFECT_POLICY_SCHEMA, operator_schema_intake_sha256=intake.sha256)
    write(options["component_coverage"], policy)
    options["operator_schema_intake"] = intake
    return options


def test_actual_normal_bounded_generation_selects_complete_logical_alias_source(effect_generation):
    report = automatic_fixtures.run(effect_generation)
    A.verify(report["automatic_derivation"], report=report)
    assert report["status"] == "incomplete"
    rows = [row for row in report["obligations"] if row["id"].startswith("auto_may_alias_result_")]
    assert {row["cohort"] for row in rows} == {"functional_guard", "withheld_transfer"}
    for row in rows:
        assert row["state"] == "generated"
        for member in row["members"]:
            root = effect_generation["output_root"] / member["member"]
            capsule = yaml.safe_load((root / "capsule.yaml").read_bytes())
            source = (root / capsule["linalg_mlir"]).read_text()
            assert "linalg.copy" in source
            assert "return %A, %A," in source
            leaves = materialize_capsule_leaves(capsule)
            from merlin.targetgen import golden_store

            expected = golden_store.load_golden(root)["outputs"]
            assert set(expected) == {"Yinput", "Yview", "Ycopy"}
            a = leaves["A"]
            independently_read_input = [
                [int(a.data[i * a.shape[1] + j]) for j in range(a.shape[1])] for i in range(a.shape[0])
            ]
            for output in expected.values():
                assert output == independently_read_input
            assert max(leaves["A"].shape) <= 3
    assert ("physical_effect", "may_alias_result") in {
        (row["kind"], row["selector"]) for row in report["automatic_derivation"]["required_unknowns"]
    }
    assert any(
        row["selector"] == "original_operator_effects" for row in report["automatic_derivation"]["required_unknowns"]
    )


def test_saved_or_changed_schema_records_cannot_recreate_live_authority(effect_generation):
    intake = effect_generation["operator_schema_intake"]
    forged = O.IndependentOperatorSchemaIntake(intake.software, intake.source_pins, intake.receipt_json)
    with pytest.raises(RtlIntakeRefusal, match="live independent issuance"):
        forged.verify()
    native = next(
        pin
        for pin in intake.source_pins
        if pin.role == "operator-schema-native-evidence" and Path(pin.path).name.startswith("observation-")
    )
    Path(native.path).write_text("{}")
    with pytest.raises(RtlIntakeRefusal, match="source changed"):
        intake.verify()


def test_versioned_effect_policy_requires_actual_live_schema_input(effect_generation):
    effect_generation.pop("operator_schema_intake")
    with pytest.raises(ValueError, match="identical live independent schema"):
        automatic_fixtures.generation.generate_target("fixture", **effect_generation)
