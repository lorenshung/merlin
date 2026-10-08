"""Single-call capsules keep private answers and reject incomplete boundary evidence."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from merlin.targetgen import capsule_common
from merlin.targetgen import core_aten_capsules as C
from merlin.targetgen.core_aten_capture import case_capture_name


@pytest.fixture
def packaged(tmp_path):
    tensor = dict(
        kind="tensor", dtype="float32", shape=[2], stride=[1], storage_offset=0, requires_grad=False, values=[1.0, 2.0]
    )
    arguments = dict(args=dict(kind="tuple", items=[tensor]), kwargs={})
    case = dict(
        overload="aten.alias.default",
        case_id="heldout alias case",
        arguments=arguments,
        post_arguments=copy.deepcopy(arguments),
        expected=tensor,
        comparison="torch_close",
        comparison_parameters={"rtol": 0, "atol": 0},
        mutated_arguments=[],
        output_input_aliases=["result->args[0]"],
    )
    source = tmp_path / "captures" / case_capture_name(case)
    source.mkdir(parents=True)
    meta = {key: value for key, value in tensor.items() if key not in {"kind", "values"}}
    capture = dict(
        overload=case["overload"],
        status="captured_exact",
        capture_meta=dict(
            ok=True,
            opaque=0,
            input_abi=[dict(dtype="f32", shape=[2])],
            output_abi=[dict(dtype="f32", shape=[2])],
            result_contract=dict(
                authority="same_conversion_exported_program",
                inputs=[meta],
                results=[dict(meta, role="user_output", alias_inputs=[0])],
            ),
        ),
    )
    (source / "capture.json").write_text(json.dumps(capture))
    (source / "inputs.json").write_text("[[1.0, 2.0]]")
    (source / "golden.json").write_text('{"private_answer": 987654321}')
    (source / "capsule.pytorch.py").write_text("# input loader fixture\n")
    (source / "capsule.linalg.mlir").write_text(
        "builtin.module {\nfunc.func @forward(%x: tensor<2xf32>) -> tensor<2xf32> {\n"
        '%y = "tensor.cast"(%x) {prov.aten = "aten.alias.default"} : (tensor<2xf32>) -> tensor<2xf32>\n'
        "func.return %y : tensor<2xf32>\n}\n}\n"
    )
    public, private = tmp_path / "public", tmp_path / "_private" / "public"
    manifest = C.write_capsules({"cases": [case]}, source.parent, public, private)
    capsule = capsule_common.load_capsule(public / manifest["capsules"][0]["id"])
    raw = [np.array([1, 2], np.float32).tobytes()] * 2
    readback = dict(metadata=[meta, meta], aliases=[[1], [1]], pre_bytes={"0": raw[1].hex()})
    return capsule, raw, readback, public, private


def test_writer_private_answers_and_schema_safe_additive_id(packaged):
    capsule, _, _, public, private = packaged
    assert capsule["name"].startswith("case_")
    assert "987654321" not in "".join(p.read_text() for p in public.rglob("*") if p.is_file() and p.suffix != ".npz")
    assert not list(public.rglob("golden*"))
    assert private.stat().st_mode & 0o777 == 0o700
    assert (private / capsule["name"] / "golden.yaml").stat().st_mode & 0o777 == 0o600
    assert C.load_private(capsule)["batch_map"]["output_count"] == 2


def test_writer_refuses_lost_exact_overload_provenance(packaged):
    capsule, _, _, public, _ = packaged
    case = C.load_private(capsule)["batch_map"]["cases"][0]["case"]
    captures = public.parent / "captures"
    source = captures / case_capture_name(case) / "capsule.linalg.mlir"
    source.write_text(source.read_text().replace("aten.alias.default", "aten.clone.default"))
    with pytest.raises(ValueError, match="exact overload provenance"):
        C.write_capsules({"cases": [case]}, captures, public.parent / "new-public", public.parent / "new-private")


@pytest.mark.parametrize("defect", ["missing_readback", "alias", "output", "post", "snapshot"])
def test_full_call_rejects_independent_boundary_defects(packaged, defect):
    capsule, raw, readback, _, _ = packaged
    assert C.grade_readback(capsule, raw, semantic_readback=readback, provenance={"scope": "test"})["passed_count"] == 1
    if defect == "missing_readback":
        readback = None
    elif defect == "alias":
        readback["aliases"][0] = []
    elif defect in {"output", "post"}:
        raw[0 if defect == "output" else 1] = np.zeros(2, np.float32).tobytes()
    else:
        readback["pre_bytes"] = {}
    assert C.grade_readback(capsule, raw, semantic_readback=readback, provenance={"scope": "test"})["passed_count"] == 0


@pytest.mark.parametrize("surface", ["capsule.interface.mlir", "inputs.npz", "golden.yaml"])
def test_bound_inputs_and_answers_fail_closed(packaged, surface):
    capsule, _, _, public, private = packaged
    root = private if surface == "golden.yaml" else public
    path = root / capsule["name"] / surface
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed"):
        C.load_private(capsule)


def test_readback_preserves_round2_result_count_and_absent_shape_frames(packaged):
    capsule, raw, readback, _, _ = packaged
    assert (
        C.grade_readback(capsule, raw, output_shapes=[], semantic_readback=readback, provenance={"scope": "test"})[
            "passed_count"
        ]
        == 1
    )
    assert (
        C.grade_readback(capsule, raw + raw, semantic_readback=readback, provenance={"scope": "test"})["passed_count"]
        == 0
    )
    assert (
        C.grade_readback(capsule, raw, output_shapes=[[2]], semantic_readback=readback, provenance={"scope": "test"})[
            "passed_count"
        ]
        == 0
    )


def test_runner_grades_candidate_emission_and_requires_rtl(packaged, tmp_path, monkeypatch):
    capsule, raw, readback, _, _ = packaged
    config = SimpleNamespace(target="test_target", fourth_output_name="lowered.mlir", rtl_tiers={"L3"})
    paths = SimpleNamespace(run_path=tmp_path / "runs")
    seen = []

    def entrypoints(*args, **kwargs):
        return None, {}, "submitted LLVM"

    def execute(**kwargs):
        seen.append(kwargs["llvm_mlir"])
        return dict(
            output_bytes=raw,
            semantic_readback=readback,
            provenance={"scope": "test"},
            engine="test",
            derived_from_rtl=False,
        )

    monkeypatch.setattr(capsule_common, "run_entrypoints", entrypoints)
    adapter = SimpleNamespace(run_full_call=execute)
    options = dict(paths=paths, config=config, pkg=None, contract=None, timeout=1, no_oracle=False)
    result = C.run_capsule(capsule, tmp_path, adapters={"L2": adapter}, **options)
    assert result["status"] == "pass" and seen == ["submitted LLVM"]
    raw[0] = np.zeros(2, np.float32).tobytes()
    assert C.run_capsule(capsule, tmp_path, adapters={"L2": adapter}, **options)["status"] == "fail"
    raw[0] = raw[1]
    capsule["required_oracle_tiers"] = ["L2", "L3"]
    assert (
        C.run_capsule(capsule, tmp_path, adapters={"L2": adapter, "L3": adapter}, **options)["status"] == "incomplete"
    )
    capsule["required_oracle_tiers"] = ["L2"]
    assert C.run_capsule(capsule, tmp_path, adapters={"L2": object()}, **options)["status"] == "incomplete"


def test_source_bundle_scoring_requires_device_evidence_and_reports_guards(packaged, tmp_path, monkeypatch):
    capsule, raw, readback, _, _ = packaged
    capsule.update(cohort="public", scored=True, lane_expectation="device")
    config = SimpleNamespace(target="test_target", fourth_output_name="lowered.mlir", rtl_tiers={"L3"})
    options = dict(
        paths=SimpleNamespace(run_path=tmp_path / "runs"),
        config=config,
        pkg=None,
        contract=None,
        timeout=1,
        no_oracle=False,
    )
    observation = dict(
        output_bytes=raw,
        semantic_readback=readback,
        provenance={"scope": "test"},
        engine="test",
        derived_from_rtl=False,
        lane="device",
        executed_instructions=0,
    )

    def execute(**kwargs):
        assert kwargs["package_dir"] == tmp_path
        assert kwargs["capsule"] is capsule
        return observation

    adapter = SimpleNamespace(run_full_call=execute, compiles_source_bundle=True)
    assert C.run_capsule(capsule, tmp_path, adapters={"L2": adapter}, **options)["status"] == "fail"
    observation["executed_instructions"] = 3
    # A bare counter cannot establish retired instruction evidence.
    assert C.run_capsule(capsule, tmp_path, adapters={"L2": adapter}, **options)["status"] == "fail"
    observation["execution_evidence"] = {"target": "test_target", "executed_instructions": 3}
    monkeypatch.setattr(
        "merlin.targetgen.core_aten_device.verify_execution_evidence",
        lambda e: e == {"target": "test_target", "executed_instructions": 3},
    )
    result = C.run_capsule(capsule, tmp_path, adapters={"L2": adapter}, **options)
    assert result["status"] == "pass"
    assert (result["lane"], result["executed_instructions"], result["cohort"]) == ("device", 3, "public")
    observation["execution_evidence"]["target"] = "foreign_target"
    assert C.run_capsule(capsule, tmp_path, adapters={"L2": adapter}, **options)["status"] == "fail"
    observation["execution_evidence"].update(target="test_target", executed_instructions=2)
    assert C.run_capsule(capsule, tmp_path, adapters={"L2": adapter}, **options)["status"] == "fail"
    observation["execution_evidence"]["executed_instructions"] = 3
    capsule.update(cohort="host_guard", scored=False, lane_expectation="host-guard")
    assert C.run_capsule(capsule, tmp_path, adapters={"L2": adapter}, **options)["status"] == "fail"
    observation.update(lane="host", executed_instructions=0)
    result = C.run_capsule(capsule, tmp_path, adapters={"L2": adapter}, **options)
    assert result["status"] == "pass" and result["cohort"] == "guard" and result["scored"] is False


def test_scored_host_expectation_enforces_host_lane(packaged, tmp_path):
    capsule, raw, readback, _, _ = packaged
    capsule.update(cohort="public", scored=True, lane_expectation="host")
    observation = dict(
        output_bytes=raw,
        semantic_readback=readback,
        provenance={"scope": "test"},
        engine="test",
        derived_from_rtl=False,
        lane="device",
        executed_instructions=1,
    )
    adapter = SimpleNamespace(run_full_call=lambda **kw: observation, compiles_source_bundle=True)
    options = dict(
        paths=SimpleNamespace(run_path=tmp_path / "runs"),
        config=SimpleNamespace(target="test_target", fourth_output_name="lowered.mlir", rtl_tiers={"L3"}),
        pkg=None,
        contract=None,
        timeout=1,
        no_oracle=False,
    )
    assert C.run_capsule(capsule, tmp_path, adapters={"L2": adapter}, **options)["status"] == "fail"
    observation.update(lane="host", executed_instructions=0)
    result = C.run_capsule(capsule, tmp_path, adapters={"L2": adapter}, **options)
    assert result["status"] == "pass" and result["scored"] is True


def test_two_snapshot_backends_grade_consecutively_in_one_process(packaged, tmp_path, monkeypatch):
    from merlin.common.paths import repo_root
    from merlin.llvmlower import toolchain
    from merlin.targetgen.plugins import load_module

    provider = load_module(
        repo_root() / "examples" / "gemmini_fp32" / "phase0" / "core_aten",
        "execution_provider.py",
        package_name="test_snapshot_execution_provider",
    )
    monkeypatch.setattr(toolchain, "llvm_install", lambda: tmp_path)
    capsule, raw, readback, _, _ = packaged
    packages = []
    for marker in ("first", "second"):
        package = tmp_path / marker / "submission"
        module = package / "mlir_oot"
        module.mkdir(parents=True)
        (module / "identity.py").write_text(f"MARKER = {marker!r}\n")
        (module / "golden_device_catalog.py").write_text(
            "from .identity import MARKER\n"
            "def build_catalog(source):\n"
            "    return None, {'covered_contractions': [MARKER]}\n"
            "def merlin_builder(tools):\n"
            "    return MARKER\n"
        )
        packages.append(package)
    loaded = []

    def execute(*, bundle, package_dir, **kwargs):
        routing = provider.routing(
            bundle,
            target="test_target",
            package=package_dir,
            facts={"facts": {"datapaths": [{"name": "input", "dtype": "i8"}, {"name": "accumulator", "dtype": "i32"}]}},
            eligible=[(None, SimpleNamespace(dtypes=("i8", "i8", "i32")))],
        )
        loaded.append(routing.catalog_builder)
        return dict(output_bytes=raw, semantic_readback=readback, provenance={"scope": "test"})

    config = SimpleNamespace(target="test_target", fourth_output_name="lowered.mlir", rtl_tiers=set())
    adapter = SimpleNamespace(run_full_call=execute, compiles_source_bundle=True)
    # Revisit the first snapshot too: cached imports must retain their original identity.
    for index, package in enumerate([*packages, packages[0]]):
        result = C.run_capsule(
            capsule,
            package,
            paths=SimpleNamespace(run_path=tmp_path / f"grade-{index}"),
            config=config,
            adapters={"L2": adapter},
            pkg=None,
            contract=None,
            timeout=1,
            no_oracle=False,
        )
        assert result["status"] == "pass", result["failure"]
    assert loaded == ["first", "second", "first"]


@pytest.mark.parametrize(
    "error", ["PluginError: two packages claim the namespace 'backend'", "ValueError: build refused", None]
)
def test_execution_refusal_preserves_tier_reason_and_compile_category(packaged, tmp_path, error):
    from merlin.targetgen.tier_integrity import reason_for

    capsule, _, _, _, _ = packaged
    adapter = SimpleNamespace(
        compiles_source_bundle=True,
        run_full_call=lambda **_: dict(output_bytes=None, execution_error=error, provenance={"scope": "test"}),
    )
    result = C.run_capsule(
        capsule,
        tmp_path,
        paths=SimpleNamespace(run_path=tmp_path / "grade"),
        config=SimpleNamespace(target="test_target", fourth_output_name="lowered.mlir", rtl_tiers=set()),
        adapters={"L2": adapter},
        pkg=None,
        contract=None,
        timeout=1,
        no_oracle=False,
    )
    reason = error or "no hardware output"
    assert result["status"] == "fail"
    assert result["failure"] == dict(plane="compile", category="PROTOCOL_VIOLATION", tier="L2", detail=reason)
    assert result["numeric"]["status"] == "skipped"
    assert result["tiers"]["L2"]["reason"] == reason
    assert reason_for("L2", result["tiers"]["L2"]) == reason
    assert json.loads((tmp_path / "grade" / "capsule_result.json").read_text())["tiers"]["L2"]["reason"] == reason
