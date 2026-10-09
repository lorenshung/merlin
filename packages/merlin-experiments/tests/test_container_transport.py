"""Closed source/image/control tests; optional real delegated namespace controls.

Native controls require explicit owned artifact/tool selections. They neither
issue fresh compiler origin nor numerical/physical/service authorities.
"""

import hashlib
import io
import json
import os
import subprocess
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest
from merlin_experiments.execution import container_image as I
from merlin_experiments.execution import container_transport as T
from merlin_experiments.execution.container_policy import verify_owned_container
from merlin_experiments.phase2.contracts import StageGateError


def synthetic_image(tmp_path, *, defect=None):
    layer = b"private source-fixture bytes, no target/compiler/semantic authority"
    config = json.dumps(
        {"rootfs": {"type": "layers", "diff_ids": ["sha256:" + hashlib.sha256(layer).hexdigest()]}}
    ).encode()
    original, archive = tmp_path / "config.json", tmp_path / "image.tar"
    original.write_bytes(config)
    manifest = [{"Config": "config.json", "Layers": [] if defect == "missing-layer" else ["layer.tar"]}]
    files = {
        "manifest.json": json.dumps(manifest).encode(),
        "config.json": config,
        "layer.tar": b"changed" if defect == "wrong-layer" else layer,
    }
    if defect == "wrong-config":
        files["config.json"] = b"{}"
    with tarfile.open(archive, "w") as stream:
        for name, payload in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            stream.addfile(member, io.BytesIO(payload))
        if defect == "duplicate":
            stream.addfile(member, io.BytesIO(payload))
    return original, archive, hashlib.sha256(config).hexdigest()


def test_exact_export_joins_independent_config_and_every_original_layer(tmp_path):
    original, archive, digest = synthetic_image(tmp_path)
    report = I.verify_image_export(configuration=original, archive=archive, config_sha256=digest)
    assert report["image_id"] == "sha256:" + digest
    assert len(report["layers"]) == 1 and report["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert "no source or runtime authority" in report["scope"]


@pytest.mark.parametrize("defect", ["wrong-layer", "missing-layer", "wrong-config", "duplicate"])
def test_engine_metadata_and_image_names_cannot_replace_original_byte_join(tmp_path, defect):
    original, archive, digest = synthetic_image(tmp_path, defect=defect)
    with pytest.raises(ValueError):
        I.verify_image_export(configuration=original, archive=archive, config_sha256=digest)


@pytest.mark.parametrize("defect", [None, "root-file", "escaping-alias", "whiteout"])
def test_image_scope_is_derived_from_original_layer_paths_and_refuses_unmaskable_entries(tmp_path, defect):
    layer = io.BytesIO()
    with tarfile.open(fileobj=layer, mode="w") as filesystem:
        for name in ("usr", "etc", "private-public-directory", "proc", "tmp"):
            member = tarfile.TarInfo(name)
            member.type = tarfile.DIRTYPE
            filesystem.addfile(member)
        alias = tarfile.TarInfo("bin")
        alias.type = tarfile.SYMTYPE
        alias.linkname = "../escape" if defect == "escaping-alias" else "usr/bin"
        filesystem.addfile(alias)
        if defect in {"root-file", "whiteout"}:
            filesystem.addfile(tarfile.TarInfo("ungranted-root-file" if defect == "root-file" else ".wh.deleted"))
    archive = tmp_path / "root-layer.tar"
    with tarfile.open(archive, "w") as image:
        data = json.dumps([{"Layers": ["layer.tar"]}]).encode()
        for name, value in (("manifest.json", data), ("layer.tar", layer.getvalue())):
            member = tarfile.TarInfo(name)
            member.size = len(value)
            image.addfile(member, io.BytesIO(value))
    if defect:
        with pytest.raises(ValueError):
            I.image_root_masks(archive)
    else:
        assert I.image_root_masks(archive) == ("/etc", "/private-public-directory", "/usr")


@pytest.mark.parametrize("defect", ["identity", "name", "image", "token"])
def test_cleanup_refuses_a_foreign_or_unsupported_service_identity(defect):
    row = {
        "Id": "a" * 64,
        "Name": "/merlin-owned-private",
        "Image": "sha256:" + "b" * 64,
        "Config": {"Labels": {"merlin.command-owner": "private"}},
    }
    if defect == "identity":
        row["Id"] = "foreign-job"
    elif defect == "name":
        row["Name"] = "/another-owned-service-job"
    elif defect == "image":
        row["Image"] = "sha256:" + "c" * 64
    else:
        row["Config"]["Labels"]["merlin.command-owner"] = "foreign"
    with pytest.raises(StageGateError, match="exact created container"):
        verify_owned_container(row, name="merlin-owned-private", token="private", image_id="sha256:" + "b" * 64)


@pytest.mark.parametrize("defect", ["linked", "socket", "comma", "changed"])
def test_mount_grammar_refuses_unclosed_linked_special_or_changed_sources(tmp_path, defect):
    source = tmp_path / "source"
    source.write_text("owned source")
    if defect == "linked":
        linked = tmp_path / "linked"
        linked.symlink_to(source)
        source = linked
    elif defect == "socket":
        import socket

        source.unlink()
        # UNIX socket addresses have a short kernel limit. The owned directory
        # FD supplies the same actual file under arbitrarily deep test roots.
        directory = os.open(source.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(f"/proc/self/fd/{directory}/{source.name}")
                with pytest.raises(StageGateError):
                    T.mount_from_source(source, "/run/service.sock", read_only=True)
        finally:
            os.close(directory)
        return
    destination = "/bad,readonly=false" if defect == "comma" else "/candidate"
    if defect == "changed":
        mount = T.mount_from_source(source, destination, read_only=True)
        source.write_text("changed by separate source owner")
        with pytest.raises(StageGateError):
            mount.verify()
    else:
        with pytest.raises(StageGateError):
            T.mount_from_source(source, destination, read_only=True)


def original_policy(tmp_path):
    source = tmp_path / "candidate"
    source.mkdir()
    return (
        "/owned/selected-sandbox",
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
        "--clearenv",
        "--setenv",
        "HOME",
        "/tmp",
        "--setenv",
        "PATH",
        "/usr/bin:/bin",
        "--setenv",
        "PYTHONPATH",
        "/component-inputs/compiler",
        "--setenv",
        "PYTHONDONTWRITEBYTECODE",
        "1",
        "--setenv",
        "PYTHONNOUSERSITE",
        "1",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--bind",
        str(source),
        "/candidate",
        "--chdir",
        "/candidate",
    )


@pytest.mark.parametrize("defect", [None, "network", "environment", "missing-scope", "missing-cwd", "duplicate-cwd"])
def test_optional_conversion_preserves_closed_original_tool_protocol(tmp_path, defect):
    argv = original_policy(tmp_path)
    if defect == "network":
        argv += ("--share-net",)
    elif defect == "environment":
        argv += ("--setenv", "PRIVATE_TOKEN", "unadmitted")
    elif defect == "missing-scope":
        argv = tuple(value for value in argv if value != "--unshare-all")
    elif defect == "missing-cwd":
        argv = argv[:-2]
    elif defect == "duplicate-cwd":
        argv += ("--chdir", "/candidate")
    if defect:
        with pytest.raises(StageGateError):
            T.mounts_from_strict_policy(argv)
    else:
        (mount,) = T.mounts_from_strict_policy(argv)
        assert mount.destination == "/candidate" and mount.read_only is False


@pytest.fixture(scope="module")
def selected_native(tmp_path_factory):
    names = (
        "MERLIN_TEST_CONTAINER_CLIENT",
        "MERLIN_TEST_CONTAINER_ENDPOINT",
        "MERLIN_TEST_CONTAINER_CONFIG",
        "MERLIN_TEST_CONTAINER_CONFIG_SHA256",
        "MERLIN_TEST_CONTAINER_ARCHIVE",
        "MERLIN_TEST_CONTAINER_CC",
    )
    values = [os.environ.get(name) for name in names]
    if not all(values):
        pytest.skip("requires explicit independent image/tool/service selections")
    client, endpoint, configuration, digest, archive, compiler = values
    root = tmp_path_factory.mktemp("native-container-owner") / "prepared"
    return T.prepare_container_transport(
        client=Path(client),
        endpoint=endpoint,
        configuration=Path(configuration),
        configuration_sha256=digest,
        image_archive=Path(archive),
        compiler=Path(compiler),
        evidence_root=root,
    ), Path(compiler)


CONTROL = r"""
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <sched.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ptrace.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
int main(int argc, char **argv) {
 if(argc!=5) return 10;
 FILE *input=fopen("/input/allowed.txt","r");char data[80]={0};
 if(!input||!fgets(data,sizeof(data),input)||strcmp(data,argv[1]))return 11;
 fclose(input);
 if(!getenv("HOME")||strcmp(getenv("HOME"),"/tmp")||getenv("DOCKER_HOST")||getenv("PRIVATE_CANARY"))return 25;
 if(open(argv[2],O_RDONLY)>=0||open("/run/docker.sock",O_RDONLY)>=0||open("/var/run/docker.sock",O_RDONLY)>=0)return 12;
 if(open("/usr/bin/sh",O_RDONLY)>=0||open("/bin/sh",O_RDONLY)>=0||open("/etc/passwd",O_RDONLY)>=0)return 26;
 if(open("/candidate/forbidden-write",O_CREAT|O_WRONLY,0600)>=0)return 13;
 if(open("/usr/bin/merlin-command-owner",O_WRONLY)>=0||open("/proc/1/fd/1",O_WRONLY)>=0)return 14;
 if(ptrace(PTRACE_ATTACH,1,0,0)==0||unshare(CLONE_NEWNS)==0)return 15;
 kill(1,SIGKILL); /* Namespace init must survive; delivery may report success. */
 if(socket(AF_INET,SOCK_RAW,IPPROTO_RAW)>=0)return 16;
 struct ifaddrs *interfaces=0;if(getifaddrs(&interfaces))return 22;
 for(struct ifaddrs *entry=interfaces;entry;entry=entry->ifa_next)
   if(!(entry->ifa_flags&IFF_LOOPBACK)){freeifaddrs(interfaces);return 24;}
 freeifaddrs(interfaces);
 if(getppid()!=1)return 17;
 FILE *output=fopen("/output/complete.txt","w");if(!output)return 18;
 fputs(data,output);fclose(output);
 if(!strcmp(argv[3],"descendant")){
   pid_t first=fork();if(first<0)return 19;
   if(!first){pid_t second=fork();if(second<0)_exit(20);if(second)_exit(0);
     setsid();signal(SIGTERM,SIG_IGN);puts("OWNED_DESCENDANT");fflush(stdout);while(1)pause();}
   if(waitpid(first,0,0)<0)return 21;usleep(100000);
 }
 if(!strcmp(argv[3],"timeout"))while(1)pause();
 if(!strcmp(argv[3],"terminate-owner")){kill(1,SIGTERM);while(1)pause();}
 if(!strcmp(argv[3],"spoof"))puts("{\"cleanup_wait_echild\":false,\"schema\":\"fake\"}");
 if(!strcmp(argv[3],"overflow")){for(int i=0;i<200000;i++)putchar('x');fflush(stdout);}
 puts("ACTUAL_PRIVATE_GRANT_CONTROL");return atoi(argv[4]);
}
"""


def native_case(tmp_path, selected_native, mode="normal", code=0, *, wrong_original=False):
    from merlin.common import invocation_record

    transport, compiler = selected_native
    candidate, out, private = (tmp_path / name for name in ("candidate", "output", "private"))
    for path in (candidate, out, private):
        path.mkdir()
    source = candidate / "control.c"
    source.write_text(CONTROL)
    tool = candidate / "control"
    invocation_record.run(
        [str(compiler), "-static", "-O2", str(source), "-o", str(tool)],
        directory=tmp_path / "build",
        stage="independent_access_control_build",
        inputs=(source,),
        outputs=(tool,),
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    original, forbidden = private / "allowed.txt", private / "forbidden.txt"
    nonce = "SOURCE_BOUND_PRIVATE_TEST_INPUT\n"
    original.write_text(nonce)
    forbidden.write_text("SYNTHETIC PRIVATE SIBLING, NOT USER CREDENTIALS")
    mounts = (
        T.mount_from_source(candidate, "/candidate", read_only=True),
        T.mount_from_source(original, "/input/allowed.txt", read_only=True),
        T.mount_from_source(out, "/output", read_only=False),
    )
    command = ("/candidate/control", "wrong original" if wrong_original else nonce, str(forbidden), mode, str(code))
    return transport, mounts, command, out, tmp_path / "execution"


@pytest.mark.parametrize("mode,code", [("normal", 0), ("descendant", 0), ("spoof", 0), ("normal", 23)])
def test_actual_container_grants_denials_and_native_owned_descendant_cleanup(tmp_path, selected_native, mode, code):
    transport, mounts, command, out, evidence = native_case(tmp_path, selected_native, mode, code)
    result = transport.execute(mounts=mounts, command=command, cwd="/candidate", evidence_root=evidence, timeout_s=3)
    assert result.returncode == code and b"ACTUAL_PRIVATE_GRANT_CONTROL" in result.stdout and not result.stderr
    assert (out / "complete.txt").read_text() == command[1]
    report = json.loads((evidence / "transport.json").read_text())
    assert report["native_cleanup"] == "COMPLETE" and report["owned_container_removal"] == "OBSERVED"
    assert report["delegated_service_authority"] == report["os_hard_deadline"] == "UNKNOWN"
    assert report["compiler_runtime_physical_roles"] == "UNISSUED"
    if mode == "descendant":
        assert report["native_result"]["reaped_after_command"] >= 1
    if mode == "spoof":
        assert b'"schema":"fake"' in result.stdout


@pytest.mark.parametrize("mode", ["timeout", "overflow"])
def test_actual_native_timeout_and_output_overflow_refuse_after_owned_cleanup(tmp_path, selected_native, mode):
    transport, mounts, command, out, evidence = native_case(tmp_path, selected_native, mode)
    with pytest.raises(StageGateError, match="bounded execution/output budget"):
        transport.execute(mounts=mounts, command=command, cwd="/candidate", evidence_root=evidence, timeout_s=0.3)
    report = json.loads((evidence / "transport.json").read_text())
    assert report["native_cleanup"] == "COMPLETE" and report["owned_container_removal"] == "OBSERVED"


def test_actual_wrong_original_bytes_keep_full_output_check_and_do_not_gain_authority(tmp_path, selected_native):
    transport, mounts, command, out, evidence = native_case(tmp_path, selected_native, wrong_original=True)
    result = transport.execute(mounts=mounts, command=command, cwd="/candidate", evidence_root=evidence, timeout_s=3)
    assert result.returncode == 11 and not (out / "complete.txt").exists()


def test_original_guardian_refuses_execution_outside_actual_pid_namespace(selected_native):
    transport, _ = selected_native
    result = subprocess.run([str(transport.guardian), "1", "--", "/usr/bin/true"], capture_output=True, timeout=3)
    assert result.returncode == 125 and not result.stdout


def test_reconstructed_preparation_cannot_enter_an_ordinary_execution_path(selected_native):
    transport, _ = selected_native
    with pytest.raises(StageGateError, match="not prepared around actual original tools"):
        replace(transport).verify()


def test_actual_partial_create_loss_recovers_and_removes_only_original_unstarted_container(
    tmp_path,
    selected_native,
    monkeypatch,
):
    transport, mounts, command, out, evidence = native_case(tmp_path, selected_native)
    original = T.invocation_record.run

    def lose_after_create(argv, **kwargs):
        result = original(argv, **kwargs)
        if kwargs["stage"] == "container_create":
            raise TimeoutError("actual create completed before private coordinator interruption")
        return result

    monkeypatch.setattr(T.invocation_record, "run", lose_after_create)
    with pytest.raises(TimeoutError, match="actual create completed"):
        transport.execute(mounts=mounts, command=command, cwd="/candidate", evidence_root=evidence, timeout_s=3)
    report = json.loads((evidence / "transport.json").read_text())
    assert report["owned_container_removal"] == "OBSERVED" and report["native_cleanup"] == "UNKNOWN"
    assert report["observed_owned_state"]["Status"] == "created"
    assert not (out / "complete.txt").exists()
    assert any("container_partial_create_inspect" in str(path) for path in evidence.rglob("invocation.json"))


def test_actual_untrusted_command_termination_of_owner_refuses_cleanup_authority(tmp_path, selected_native):
    transport, mounts, command, out, evidence = native_case(tmp_path, selected_native)
    # A trusted owner resists SIGKILL delivery from a child; its TERM handler may
    # cancel the command, which must refuse instead of publishing completion.
    request = (command[0], command[1], command[2], "terminate-owner", "0")
    with pytest.raises(StageGateError, match="bounded execution/output budget"):
        transport.execute(mounts=mounts, command=request, cwd="/candidate", evidence_root=evidence, timeout_s=3)
    report = json.loads((evidence / "transport.json").read_text())
    assert report["native_cleanup"] == "COMPLETE" and report["native_result"]["timed_out"]


ORDINARY_CONTROL = r"""
#define _GNU_SOURCE
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
int main(int argc,char **argv){
 if(argc!=4)return 31;FILE *input=fopen(argv[1],"r");char data[80]={0};
 if(!input||!fgets(data,sizeof(data),input)||strcmp(data,"original ordinary interface\n"))return 32;
 fclose(input);if(open(argv[3],O_RDONLY)>=0||open("/evaluation-input/private.txt",O_RDONLY)>=0)return 33;
 if(open("forbidden-write",O_WRONLY|O_CREAT,0600)>=0||open("/usr/bin/sh",O_RDONLY)>=0)return 34;
 pid_t first=fork();if(first<0)return 35;
 if(!first){pid_t second=fork();if(second<0)_exit(36);if(second)_exit(0);
   setsid();signal(SIGTERM,SIG_IGN);while(1)pause();}
 if(waitpid(first,0,0)<0)return 37;
 FILE *output=fopen(argv[2],"w");if(!output)return 38;
 fputs("{\"ordinary_private_input\":\"observed\"}\n",output);fclose(output);
 puts("ACTUAL_ORDINARY_PACKAGE_COMMAND");return 0;
}
"""


def test_actual_normal_package_executor_preserves_original_policy_source_and_private_outputs(
    tmp_path,
    selected_native,
    monkeypatch,
):
    from merlin_experiments.phase1 import component_package_execution as E
    from merlin_experiments.phase2.component_runtime import inventory_runtime
    from test_component_experiment import _view

    from merlin.common import invocation_record
    from merlin.targetgen import package_runtime as P

    transport, compiler = selected_native
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    source = candidate / "private-control.c"
    source.write_text(ORDINARY_CONTROL)
    tool = candidate / "private-control"
    invocation_record.run(
        [str(compiler), "-static", "-O2", str(source), "-o", str(tool)],
        directory=tmp_path / "build",
        stage="ordinary_source_control_build",
        inputs=(source,),
        outputs=(tool,),
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    evidence = tmp_path / "private" / "grade"
    evidence.mkdir(parents=True)
    original, output, private = (evidence / name for name in ("original.mlir", "output.json", "private.txt"))
    original.write_text("original ordinary interface\n")
    private.write_text("PRIVATE SYNTHETIC UNGRANTED SIBLING")
    runtime = inventory_runtime(files=(), trees=(), executables=((Path("/usr/bin/bwrap"), "/usr/bin/bwrap"),))
    package = P.Package(
        candidate,
        {"language": "c", "commands": {"parse": {"argv": ["{tool}", "{input_mlir}", "{output_json}", str(private)]}}},
        tool,
    )
    view = _view(tmp_path)
    # Only outer delegated service execution is admitted; no unboxed candidate
    # subprocess may run or discovery/fallback may occur in this optional route.
    with E.qualified_package_execution(
        candidate=candidate, view=view, runtime=runtime, evidence_root=evidence, container_transport=transport
    ):
        result = P.run_entrypoint(
            package, "parse", original, output, timeout=4, write_bytecode=False, invocation_directory=evidence
        )
    assert result.returncode == 0 and result.stdout.strip() == "ACTUAL_ORDINARY_PACKAGE_COMMAND"
    assert output.read_text() == '{"ordinary_private_input":"observed"}\n'
    assert private.read_text() == "PRIVATE SYNTHETIC UNGRANTED SIBLING"
    assert not (candidate / "forbidden-write").exists()
    records = [invocation_record.verify(path) for path in evidence.rglob("invocation.json")]
    (dispatch,) = [row for row in records if row["stage"] == "parse"]
    assert dispatch["kind"] == "python_call" and "PreparedContainerTransport.execute" in dispatch["callable"]
    assert dispatch["inputs"] == [{"path": str(original), "sha256": hashlib.sha256(original.read_bytes()).hexdigest()}]
    assert dispatch["outputs"] == [{"path": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}]
    assert any(row["stage"] == "container_command" and row["kind"] == "subprocess" for row in records)
    (report,) = [json.loads(path.read_text()) for path in evidence.rglob("transport.json")]
    assert report["native_cleanup"] == "COMPLETE" and report["native_result"]["reaped_after_command"] >= 1


def test_optional_package_route_refuses_declaration_only_transport_before_unboxed_command(tmp_path, monkeypatch):
    from merlin_experiments.phase1 import component_package_execution as E
    from merlin_experiments.phase2.component_experiment import ComponentView, RuntimeGrant

    from merlin.targetgen import package_runtime as P

    candidate = tmp_path / "candidate"
    candidate.mkdir()
    evidence = tmp_path / "grade"
    evidence.mkdir()
    source = evidence / "source.mlir"
    source.write_text("module {}")
    tool = candidate / "driver"
    tool.write_text("not executed")
    sandbox = tmp_path / "sandbox"
    sandbox.write_text("private byte fixture")
    runtime = (RuntimeGrant(sandbox, "/usr/bin/bwrap", hashlib.sha256(sandbox.read_bytes()).hexdigest()),)
    view = ComponentView(tmp_path / "view", "1" * 64, "2" * 64, "3" * 64)
    monkeypatch.setattr(
        E, "strict_tool_policy", lambda *_args, **_kwargs: (str(sandbox), "--bind", str(candidate), str(candidate))
    )
    monkeypatch.setattr(E.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("unboxed fallback"))
    package = P.Package(candidate, {"language": "c", "commands": {"parse": {"argv": ["{tool}", "{input_mlir}"]}}}, tool)
    executor = E.ComponentPackageExecutor(
        candidate, view, runtime, evidence, container_transport={"status": "qualified"}
    )
    with pytest.raises(StageGateError, match="explicitly prepared original container"):
        executor.run_entrypoint(package, "parse", source, invocation_directory=evidence)
