"""Author tool membership and real no-model isolation controls.

Private canaries and native interpreter runs grant no fresh origin, target
runtime, compiler correctness, physical equivalence or performance authority.
"""

import hashlib
import json
import os
import socket
from dataclasses import replace
from pathlib import Path

import pytest
from merlin_experiments.phase1 import component_origin as O
from merlin_experiments.phase1.providers import codex_agent as CA
from merlin_experiments.phase2.component_experiment import (
    ApprovedInput,
    RuntimeGrant,
    materialize_component_view,
    strict_tool_policy,
)
from merlin_experiments.phase2.component_runtime import inventory_runtime
from merlin_experiments.phase2.contracts import StageGateError, sha256_file

from merlin.common import invocation_record
from merlin.targetgen.compiler_library import freeze_compiler_library


def grant(path, destination):
    return RuntimeGrant(path, destination, sha256_file(path))


@pytest.fixture
def selected_tools(tmp_path):
    tool = tmp_path / "tool"
    tool.write_text("private membership control, not a compiler")
    selected = grant(tool, "/usr/bin/selected-tool")
    probe = O.FreshToolProbe("host_compiler", (selected.destination, "--version"), "1" * 64)
    return selected, probe


def test_exact_tool_membership_is_reachable_without_issuing_authority(selected_tools):
    tool, probe = selected_tools
    assert O._verify_author_tool_containment(runtime=(tool,), control_runtime=(tool,), readiness=(probe,)) is None


@pytest.mark.parametrize("change", ["missing", "source", "destination", "changed_bytes"])
def test_other_transport_readiness_cannot_replace_author_tool_membership(selected_tools, tmp_path, change):
    tool, probe = selected_tools
    if change == "missing":
        control = ()
    elif change == "destination":
        control = (replace(tool, destination="/usr/bin/other-tool"),)
    else:
        other = tmp_path / "other-tool"
        other.write_bytes(tool.source.read_bytes() if change == "source" else b"different control tool")
        control = (grant(other, tool.destination),)
    with pytest.raises(StageGateError, match="omits or changes"):
        O._verify_author_tool_containment(runtime=(tool,), control_runtime=control, readiness=(probe,))


@pytest.mark.parametrize("executable", ["selected-tool", "/usr/bin/ungranted", "/candidate/compiler"])
def test_readiness_requires_its_exact_absolute_admitted_executable(selected_tools, executable):
    tool, probe = selected_tools
    with pytest.raises(StageGateError, match="outside its admitted"):
        O._verify_author_tool_containment(
            runtime=(tool,), control_runtime=(tool,), readiness=(replace(probe, command=(executable,)),)
        )


def test_changed_and_duplicate_tool_sources_refuse_before_readiness(selected_tools):
    tool, probe = selected_tools
    with pytest.raises(StageGateError, match="duplicate"):
        O._verify_author_tool_containment(runtime=(tool,), control_runtime=(tool, tool), readiness=(probe,))
    tool.source.write_text("changed after independent selection")
    with pytest.raises(StageGateError, match="bytes changed"):
        O._verify_author_tool_containment(runtime=(tool,), control_runtime=(tool,), readiness=(probe,))


def test_saved_flags_cannot_select_author_tools(selected_tools):
    tool, probe = selected_tools
    with pytest.raises(StageGateError, match="immutable runtime grants"):
        O._verify_author_tool_containment(
            runtime=(tool,), control_runtime=({"qualified": True, "tool": tool.destination},), readiness=(probe,)
        )


_CONTROL_SOURCE = r"""#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
static int readable(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;
    close(fd); return 1;
}
int main(int argc, char **argv) {
    if (argc != 3) return 2;
    int candidate = strcmp(argv[1], "candidate") == 0;
    const char *allow[] = {"/candidate/public.txt", "/component-inputs/generated_input/public.txt"};
    for (unsigned i=0; i<2; i++) if (!readable(allow[i])) return 3;
    const char *private[] = {"/evaluator-private/private.txt", "/candidate/escape",
                            "/component-sibling/private.txt", "/tmp/.codex/auth.json",
                            "/proc/self/root/tmp/.codex/auth.json"};
    for (unsigned i=0; i<5; i++) {
        if (readable(private[i]) == candidate) { fprintf(stderr,"wrong visibility: %s\n",private[i]); return 4; }
    }
    char parent[160];
    snprintf(parent, sizeof(parent), "/proc/%ld/root/tmp/.codex/auth.json", (long)getppid());
    if (readable(parent) == candidate) return 5;
    int fd = open("/candidate/native-write.txt", O_CREAT|O_WRONLY, 0600);
    if (fd < 0) return 6;
    close(fd);
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        if (!candidate || (errno != EPERM && errno != EACCES)) return 7;
    } else {
        struct sockaddr_in address = {.sin_family=AF_INET, .sin_port=htons(atoi(argv[2])),
                                      .sin_addr.s_addr=htonl(INADDR_LOOPBACK)};
        int status = connect(sock, (struct sockaddr*)&address, sizeof(address));
        int reason = errno;
        close(sock);
        if ((!candidate && status != 0) || (candidate && (status == 0 || reason != EPERM))) return 8;
    }
    puts(candidate ? "OWNED_CANARIES_DENIED" : "OWNED_CANARIES_VISIBLE"); return 0;
}
"""


def test_actual_native_profile_denies_owned_canaries_and_reaches_selected_python(tmp_path):
    selectors = {
        name: os.environ.get(name)
        for name in (
            "MERLIN_TEST_CODEX",
            "MERLIN_TEST_BWRAP",
            "MERLIN_TEST_NATIVE_CC",
            "MERLIN_TEST_COMPONENT_PYTHON",
            "MERLIN_TEST_COMPONENT_STDLIB",
        )
    }
    if not all(selectors.values()):
        pytest.skip("requires explicit native Codex, namespace tool, C compiler and public Python/stdlib selections")
    codex, sandbox, compiler, python, stdlib = (Path(value).resolve(strict=True) for value in selectors.values())
    source, binary = tmp_path / "control.c", tmp_path / "control"
    source.write_text(_CONTROL_SOURCE)
    invocation_record.run(
        [str(compiler), "-static", str(source), "-o", str(binary)],
        directory=tmp_path / "build",
        stage="private_author_control_build",
        inputs=(source,),
        outputs=(binary,),
        timeout=30,
        check=True,
        capture_output=True,
    )
    runtime = inventory_runtime(
        files=(),
        trees=((stdlib, str(stdlib)),),
        executables=(
            (sandbox, "/usr/bin/bwrap"),
            (binary, "/usr/bin/control"),
            (python, "/usr/bin/python3"),
            *((path, str(path)) for path in sorted((stdlib / "lib-dynload").glob("*.so"))),
        ),
    )
    control_runtime = (*runtime, grant(codex, "/usr/bin/codex"))
    O._verify_author_tool_containment(runtime=runtime, control_runtime=control_runtime, readiness=())
    library = tmp_path / "library"
    library.mkdir()
    (library / "api.py").write_text('"""Diagnostic public input API, no compiler implementation."""\n')
    contract = freeze_compiler_library(
        library, review_id="owned-native-control", public_modules=("api",), sources=(("api.py", "api"),)
    )
    public = tmp_path / "public.txt"
    public.write_text("OWNED_ADMITTED_INPUT")
    view = materialize_component_view(
        tmp_path / "view",
        library=contract,
        library_root=library,
        generation_sha256="1" * 64,
        inputs=(ApprovedInput(public, "generated_input/public.txt", sha256_file(public), "generated_input"),),
    )
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "public.txt").write_text("OWNED_PUBLIC_CANARY")
    policy = list(
        strict_tool_policy(
            view, candidate, runtime=control_runtime, candidate_destination="/candidate", bwrap_binary=sandbox
        )
    )
    # Actual author control shares network. The nested native profile must
    # refuse the owned listener even when it is reachable from the control.
    policy.insert(policy.index("--unshare-all") + 1, "--share-net")
    (candidate / "escape").symlink_to("/evaluator-private/private.txt")
    config = tmp_path / "config.toml"
    config.write_text(
        CA._candidate_permission_config(
            Path("/tmp/.codex"),
            read_paths=("/component-inputs", *(row.destination for row in runtime)),
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
        Path(__file__),
        Path(CA.__file__),
        Path(O.__file__),
        Path(invocation_record.__file__),
        *(row.source for row in control_runtime),
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = str(listener.getsockname()[1])
        for name, command in (
            ("outer_control", ["/usr/bin/control", "control", port]),
            (
                "native_denials",
                [
                    "/usr/bin/codex",
                    "sandbox",
                    "--permission-profile",
                    CA._CANDIDATE_PERMISSION_PROFILE,
                    "-C",
                    "/candidate",
                    "--",
                    "/usr/bin/control",
                    "candidate",
                    port,
                ],
            ),
            (
                "admitted_python",
                [
                    "/usr/bin/codex",
                    "sandbox",
                    "--permission-profile",
                    CA._CANDIDATE_PERMISSION_PROFILE,
                    "-C",
                    "/candidate",
                    "--",
                    "/usr/bin/python3",
                    "-I",
                    "-B",
                    "-c",
                    'from pathlib import Path; Path("/candidate/python.txt").write_text('
                    'Path("/component-inputs/generated_input/public.txt").read_text()); '
                    'print("SELECTED_PUBLIC_PYTHON_READY")',
                ],
            ),
        ):
            result = invocation_record.run(
                [*policy, "--", *command],
                directory=tmp_path / name,
                stage="private_author_" + name,
                inputs=(config, private),
                dependencies=dependencies,
                timeout=45,
                check=False,
                capture_output=True,
            )
            assert result.returncode == 0, result.stderr.decode()
            expected = {
                "outer_control": b"OWNED_CANARIES_VISIBLE\n",
                "native_denials": b"OWNED_CANARIES_DENIED\n",
                "admitted_python": b"SELECTED_PUBLIC_PYTHON_READY\n",
            }[name]
            assert result.stdout == expected
    assert (candidate / "python.txt").read_text() == public.read_text()
    records = tuple(tmp_path.rglob("invocation.json"))
    assert len(records) == 4
    for record in records:
        assert invocation_record.verify(record)["returncode"] == 0
    (tmp_path / "scope.json").write_text(
        json.dumps(
            {
                "scope": "owned no-model controls; origin/runtime/compiler/physical authority UNKNOWN",
                "selectors": selectors,
                "test_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            }
        )
    )
