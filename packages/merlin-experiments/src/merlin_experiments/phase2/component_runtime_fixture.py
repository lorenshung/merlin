"""Private original-source control fixtures; absent from every author grant.

These independently declared tensor tests are evaluator mutation support, not a
seed target compiler, runtime qualification bypass or workload implementation.
"""

from pathlib import Path

from merlin_experiments.phase1.component_witness import REQUIRED_EXECUTION_EFFECTS

from . import component_runtime_controls as controls
from .component_runtime_qualification import RuntimeControlFixture
from .contracts import StageGateError, exact_tree_record, sha256_file, write_json


def prepare_source_control(*, name, root, build_service, contract_root, target_descriptor):
    mechanism, _, direction = name.partition(".")
    if root.exists():
        raise StageGateError("private runtime control needs a fresh evidence destination")
    root.mkdir(parents=True, mode=0o700)
    candidate, capsule = root / "candidate", root / "capsule"
    candidate.mkdir()
    capsule.mkdir()
    source = (
        "module { func.func @main(%a: tensor<1x3xf32>, %b: tensor<1x3xf32>) "
        "-> (tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>) { "
        "%sum = arith.addf %a, %b : tensor<1x3xf32> "
        "func.return %sum, %b, %a : tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x3xf32> } }\n"
    )
    (capsule / "source.mlir").write_text(source)
    policy = {"compare": "tolerance_float", "dtype": "f32", "atol": 0.0, "rtol": 0.0}
    write_json(root / "original_policy.json", policy)
    declaration = {
        "name": "private_runtime_control",
        "kind": "model_slice",
        "source_role": "handauthored_compiler_test",
        "label": "hidden",
        "interface_mlir": "source.mlir",
        "operation": {"op": "add"},
        "inputs": [
            {"name": tensor, "shape": [1, 3], "dtype": "f32", "role": role}
            for tensor, role in (
                ("a", "input"),
                ("b", "input"),
                ("sum", "output"),
                ("copy", "output"),
                ("identity", "output"),
            )
        ],
        "numeric_policy": policy.copy(),
        "expected": {"instruction_classes": []},
        "required_oracle_tiers": ["L2"],
    }
    if direction == "negative" and mechanism == "original_numeric_gate":
        declaration["numeric_policy"]["atol"] = 100.0
    write_json(capsule / "capsule.yaml", declaration)
    from merlin.targetgen.golden_store import write_golden

    write_golden(
        capsule,
        {
            "golden_source": "private_upstream_control",
            "outputs": {
                "sum": [[3.25, -3.0, -0.125]],
                "copy": [[2.0, 0.5, -0.25]],
                "identity": [[1.25, -3.5, 0.125]],
            },
            "oracle_provenance": {
                "inputs": {
                    "a": {"shape": [1, 3], "decoded": [1.25, -3.5, 0.125]},
                    "b": {"shape": [1, 3], "decoded": [2.0, 0.5, -0.25]},
                }
            },
        },
    )
    driver = Path(controls.__file__).read_text()
    if direction == "negative" and mechanism == "source_correspondence":
        driver = driver.replace(
            '("add", *(expressions[value] for value in op.operands))', "expressions[op.operands[0]]", 1
        )
    if direction == "negative" and mechanism == "original_output_roster":
        driver = driver.replace(
            "PrimitiveProgram(shape, len(function.body.block.args), returned)",
            "PrimitiveProgram(shape, len(function.body.block.args), returned[:-1])",
            1,
        )
    (candidate / "driver.py").write_text(driver)
    target, entry = build_service.target, build_service.recipe.require_kernel_stack_frame().entry_symbol
    commands = {
        command: {"argv": ["python3", "driver.py", command, "{input_mlir}", target, entry]}
        for command in ("parse", "lower_interface_to_target", "emit_command_buffer", "lower_target_to_llvm")
    }
    commands["emit_command_buffer"]["argv"].append("{output_json}")
    write_json(
        candidate / "manifest.yaml",
        {
            "artifact_type": "mlir_oot_target_backend",
            "target": target,
            "language": "python",
            "authoring": {"mode": "hand_curated", "scope": "private primitive transport control only"},
            "integrity_exempt": False,
            "entrypoints": {"tool": "driver.py"},
            "commands": commands,
        },
    )
    member = {
        "name": declaration["name"],
        "program_sha256": sha256_file(capsule / "source.mlir"),
        "sha256": exact_tree_record(capsule)["sha256"],
        "output_roster": ["sum", "copy", "identity"],
    }
    runs = root / "grade"
    fixture = RuntimeControlFixture(
        name,
        {
            "package_dir": candidate,
            "capsules_root": [capsule],
            "runs_root": runs,
            "contract": contract_root,
            "target": target,
            "timeout": 60,
            "max_workers": 1,
        },
        {
            "result_path": runs / declaration["name"] / "capsule_result.json",
            "member": member,
            "candidate_root": candidate,
            "compiler_snapshot": candidate,
            "candidate_sha256": exact_tree_record(candidate)["sha256"],
            "capsule_root": capsule,
            "evidence_root": root,
            "target_descriptor": target_descriptor,
            "frontend": "mlir",
            "required_effects": REQUIRED_EXECUTION_EFFECTS,
            "timeout_s": 60,
        },
        member,
        "mlir",
        exact_tree_record(candidate)["sha256"],
        sha256_file(target_descriptor),
        capsule,
        root,
        REQUIRED_EXECUTION_EFFECTS,
        (
            "component_source_lowering",
            "parse",
            "lower_interface_to_target",
            "emit_command_buffer",
            "emit_target_artifact",
        ),
    )
    return fixture
