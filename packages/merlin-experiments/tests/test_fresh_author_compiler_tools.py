"""Source-bound freestanding build inside the actual no-model author boundary.

The linked artifact is never executed. These controls establish only tool
reachability and namespace denials, not ISA, ABI, fresh origin or target runtime.
"""

import importlib.util
import json
import os
import socket
from pathlib import Path

import pytest
from merlin_experiments.phase1 import component_origin as O
from merlin_experiments.phase1.providers import codex_agent as CA
from merlin_experiments.phase2 import component_experiment as E
from merlin_experiments.phase2 import component_runtime as R
from merlin_experiments.phase2.contracts import sha256_file

from merlin.common import invocation_record
from merlin.targetgen.compiler_library import freeze_compiler_library

_SOURCE = """module {
 llvm.func @author_control(%a: i32, %b: i32) -> i32 {
  %c = llvm.add %a, %b : i32
  llvm.return %c : i32
 }
}
"""


def _selected():
    names = (
        "MERLIN_TEST_CODEX",
        "MERLIN_TEST_CODEX_BWRAP",
        "MERLIN_TEST_BWRAP",
        "MERLIN_TEST_NATIVE_CC",
        "MERLIN_TEST_MLIR_TRANSLATE",
        "MERLIN_TEST_CLANG",
        "MERLIN_TEST_CROSS_GCC",
        "MERLIN_TEST_TOOLCHAIN_PREFIX",
        "MERLIN_TEST_TOOLCHAIN_LIB_PREFIX",
    )
    values = {name: os.environ.get(name) for name in names}
    if not all(values.values()):
        pytest.skip("requires explicit native author, translator, compiler, driver and sparse relocation prefix")
    paths = tuple(Path(value) for value in values.values())
    assert all(path.is_absolute() for path in paths)
    return values, tuple(path.resolve(strict=True) for path in paths)


def _probe(root, name, command, dependencies):
    result = invocation_record.run(
        command,
        directory=root / name,
        stage="private_author_" + name,
        dependencies=dependencies,
        timeout=30,
        check=True,
        capture_output=True,
    )
    return result.stdout.decode().strip()


def _private_control_source():
    # Explicit file loading also works for wheel tests using importlib mode;
    # this shared private C canary is never an experiment input or compiler seed.
    path = Path(__file__).with_name("test_fresh_author_tools.py")
    spec = importlib.util.spec_from_file_location("private_author_denial_canary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return path, module._CONTROL_SOURCE


def test_actual_source_translates_compiles_and_links_inside_native_author_boundary(tmp_path):
    selectors, (codex, nested_sandbox, sandbox, native_cc, translator, clang, driver, prefix, library_prefix) = (
        _selected()
    )
    assert driver.is_relative_to(prefix)
    destination_prefix = Path("/usr/author-toolchain")
    clang_destination = str(clang)
    driver_destination = str(destination_prefix / driver.relative_to(prefix))
    target = _probe(tmp_path, "driver_target", [str(driver), "-dumpmachine"], (driver,))
    assert target and not any(character.isspace() for character in target)
    helpers = []
    helper_destinations = {}
    for name in ("cc1", "as", "ld", "collect2"):
        raw = _probe(tmp_path, "driver_" + name, [str(driver), "-print-prog-name=" + name], (driver,))
        path = Path(raw).resolve(strict=True)
        assert Path(raw).is_absolute() and path.is_relative_to(prefix)
        helper_destinations[name] = str(destination_prefix / path.relative_to(prefix))
        helpers.append((path, helper_destinations[name]))
    linker_search = "-B" + str(Path(helper_destinations["ld"]).parent) + "/"
    canary_fixture, control_source = _private_control_source()
    source, binary = tmp_path / "denials.c", tmp_path / "denials"
    source.write_text(control_source)
    invocation_record.run(
        [str(native_cc), "-static", str(source), "-o", str(binary)],
        directory=tmp_path / "control_build",
        stage="private_author_control_build",
        inputs=(source,),
        outputs=(binary,),
        timeout=30,
        check=True,
        capture_output=True,
    )
    selected = (
        (nested_sandbox, "/usr/bin/bwrap"),
        (binary, "/usr/bin/denials"),
        (translator, "/usr/bin/mlir-translate"),
        (clang, clang_destination),
        (driver, driver_destination),
        *helpers,
    )
    for index, (path, _) in enumerate(selected):
        result = invocation_record.run(
            ["/usr/bin/ldd", str(path)],
            directory=tmp_path / ("loader_" + str(index)),
            stage="private_author_loader_dependencies",
            dependencies=(path,),
            timeout=30,
            check=False,
            capture_output=True,
        )
        assert result.returncode == 0 or b"not a dynamic executable" in result.stderr + result.stdout
    runtime = R.inventory_runtime(
        files=(),
        trees=(),
        executables=selected,
        dependency_relocations=((prefix, str(destination_prefix)), (library_prefix, "/usr/lib")),
    )
    control_runtime = (*runtime, E.RuntimeGrant(codex, "/usr/bin/codex", sha256_file(codex)))
    # Exact membership validation only; readiness is observed by the real
    # connected invocations below, never by a synthetic FreshToolProbe receipt.
    O._verify_author_tool_containment(runtime=runtime, control_runtime=control_runtime, readiness=())
    assert all(row.source.is_file() for row in runtime)
    library = tmp_path / "library"
    library.mkdir()
    (library / "api.py").write_text('"""Private input canary, no compiler implementation."""\n')
    contract = freeze_compiler_library(
        library, review_id="owned-build-control", public_modules=("api",), sources=(("api.py", "api"),)
    )
    original = tmp_path / "source.mlir"
    original.write_text(_SOURCE)
    public = tmp_path / "public.txt"
    public.write_text("OWNED_ADMITTED_INPUT")
    view = E.materialize_component_view(
        tmp_path / "view",
        library=contract,
        library_root=library,
        generation_sha256="1" * 64,
        inputs=(
            E.ApprovedInput(original, "generated_input/source.mlir", sha256_file(original), "generated_input"),
            E.ApprovedInput(public, "generated_input/public.txt", sha256_file(public), "generated_input"),
        ),
    )
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "public.txt").write_text("OWNED_PUBLIC_CANARY")
    policy = list(
        E.strict_tool_policy(
            view, candidate, runtime=control_runtime, candidate_destination="/candidate", bwrap_binary=sandbox
        )
    )
    policy.insert(policy.index("--unshare-all") + 1, "--share-net")
    (candidate / "escape").symlink_to("/evaluator-private/private.txt")
    config = tmp_path / "config.toml"
    config.write_text(
        CA._candidate_permission_config(
            Path("/tmp/.codex"), read_paths=("/component-inputs", *(row.destination for row in runtime))
        )
    )
    private = tmp_path / "private.txt"
    private.write_text("OWNED_PRIVATE_CANARY; no credential or validation input")
    for destination in ("/evaluator-private/private.txt", "/component-sibling/private.txt", "/tmp/.codex/auth.json"):
        policy += ["--ro-bind", str(private), destination]
    policy += ["--ro-bind", str(config), "/tmp/.codex/config.toml"]
    dependencies = (
        source,
        config,
        private,
        original,
        public,
        canary_fixture,
        Path(__file__),
        Path(CA.__file__),
        Path(O.__file__),
        Path(E.__file__),
        Path(R.__file__),
        Path(invocation_record.__file__),
        *(row.source for row in control_runtime),
    )

    def native(name, command, inputs=(), outputs=()):
        result = invocation_record.run(
            [
                *policy,
                "--",
                "/usr/bin/codex",
                "sandbox",
                "--permission-profile",
                CA._CANDIDATE_PERMISSION_PROFILE,
                "-C",
                "/candidate",
                "--",
                *command,
            ],
            directory=tmp_path / name,
            stage="private_author_" + name,
            inputs=inputs,
            outputs=outputs,
            dependencies=dependencies,
            timeout=45,
            check=False,
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr.decode()
        return result

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = str(listener.getsockname()[1])
        outer = invocation_record.run(
            [*policy, "--", "/usr/bin/denials", "control", port],
            directory=tmp_path / "outer_control",
            stage="private_author_outer_control",
            dependencies=dependencies,
            timeout=45,
            check=True,
            capture_output=True,
        )
        assert outer.stdout == b"OWNED_CANARIES_VISIBLE\n"
        assert (
            native("denied_before_build", ["/usr/bin/denials", "candidate", port]).stdout == b"OWNED_CANARIES_DENIED\n"
        )
        llvm, object_file, linked = (candidate / name for name in ("source.ll", "source.o", "source.elf"))
        native(
            "translation",
            [
                "/usr/bin/mlir-translate",
                "--mlir-to-llvmir",
                "/component-inputs/generated_input/source.mlir",
                "-o",
                "/candidate/source.ll",
            ],
            (original,),
            (llvm,),
        )
        native(
            "object",
            [clang_destination, "--target=" + target, "-c", "/candidate/source.ll", "-o", "/candidate/source.o"],
            (llvm,),
            (object_file,),
        )
        assert (
            native("relocated_linker_selection", [driver_destination, linker_search, "-print-prog-name=ld"])
            .stdout.decode()
            .strip()
            == helper_destinations["ld"]
        )
        native(
            "link",
            [
                driver_destination,
                linker_search,
                "-nostdlib",
                "-fno-use-linker-plugin",
                "-Wl,-e,author_control",
                "/candidate/source.o",
                "-o",
                "/candidate/source.elf",
            ],
            (object_file,),
            (linked,),
        )
        assert (
            native("denied_after_build", ["/usr/bin/denials", "candidate", port]).stdout == b"OWNED_CANARIES_DENIED\n"
        )
    assert linked.read_bytes().startswith(b"\x7fELF") and object_file.read_bytes().startswith(b"\x7fELF")
    assert "define" in llvm.read_text() and "@author_control" in llvm.read_text()
    records = tuple(tmp_path.rglob("invocation.json"))
    for record in records:
        observation = json.loads(record.read_text())
        if observation["returncode"] != 0:
            # The original known-static, actually executed private canary has
            # no loader. Reopen its exact ldd refusal without declaring a failed
            # compiler invocation successful or changing the recorded status.
            assert observation["stage"] == "private_author_loader_dependencies"
            assert observation["argv"] == ["/usr/bin/ldd", str(binary)]
            assert observation["returncode"] == 1 and observation["status"] == "failed"
            assert all(
                observation[name] for name in ("inputs_unchanged", "dependencies_unchanged", "executable_unchanged")
            )
            assert b"not a dynamic executable" in (
                Path(observation["stdout"]["path"]).read_bytes() + Path(observation["stderr"]["path"]).read_bytes()
            )
            for pin in (
                observation["executable"],
                observation["stdout"],
                observation["stderr"],
                *observation["inputs"],
                *observation["outputs"],
                *observation["dependencies"],
            ):
                assert sha256_file(Path(pin["path"])) == pin["sha256"]
        else:
            invocation_record.verify(record)
    (tmp_path / "scope.json").write_text(
        json.dumps(
            {
                "scope": (
                    "freestanding scalar compile/link and owned denials only; artifact never executed; "
                    "origin/ISA/ABI/target runtime UNKNOWN"
                ),
                "selectors": selectors,
                "reported_target": target,
                "dependency_relocations": [[str(prefix), str(destination_prefix)], [str(library_prefix), "/usr/lib"]],
                "runtime": [
                    {"source": str(row.source), "destination": row.destination, "sha256": row.sha256}
                    for row in control_runtime
                ],
                "test_source_sha256": sha256_file(Path(__file__)),
                "linked_sha256": sha256_file(linked),
                "invocation_count": len(records),
            },
            indent=2,
        )
        + "\n"
    )
