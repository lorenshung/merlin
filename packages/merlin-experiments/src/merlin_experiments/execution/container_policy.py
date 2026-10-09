"""Closed data grants and exact ordinary compiler tool-policy conversion.

No raw policy or mount inventory grants authoring/runtime authority. Callers own
original independent admission. Existing transport metadata is compared only to
its actual supplied create plan, never interpreted as kernel isolation proof.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from merlin_experiments.phase2.contracts import StageGateError


def _tree(path):
    path = Path(path)
    if path.is_symlink() or not (path.is_file() or path.is_dir()):
        raise StageGateError("container mount source is absent or linked")
    rows = []
    for member in [path] if path.is_file() else sorted(path.rglob("*")):
        if member.is_symlink() or not (member.is_file() or member.is_dir()):
            raise StageGateError("container mount contains linked or special files")
        if member.is_file():
            rows.append(
                (
                    str(member.relative_to(path)) if path.is_dir() else ".",
                    hashlib.sha256(member.read_bytes()).hexdigest(),
                )
            )
        else:
            rows.append((str(member.relative_to(path)), "directory"))
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class ContainerMount:
    source: Path
    destination: str
    read_only: bool
    membership_sha256: str

    def verify(self):
        target = PurePosixPath(self.destination)
        if (
            not self.source.is_absolute()
            or not target.is_absolute()
            or str(target) != self.destination
            or target == PurePosixPath("/")
            or ".." in target.parts
            or any(char in str(self.source) + self.destination for char in ",\n\r\x00")
            or type(self.read_only) is not bool
            or _tree(self.source) != self.membership_sha256
        ):
            raise StageGateError("container grant differs from its closed original mount")


def mount_from_source(source: Path, destination: str, *, read_only: bool) -> ContainerMount:
    """Pin explicit data only. Callers retain their ordinary grant admission."""
    if Path(source).is_symlink():
        raise StageGateError("container grant cannot resolve a linked owner")
    actual = Path(source).resolve(strict=True)
    result = ContainerMount(actual, destination, read_only, _tree(actual))
    result.verify()
    return result


def mounts_from_strict_policy(argv: tuple[str, ...]) -> tuple[ContainerMount, ...]:
    """Consume the exact current component tool policy language, without executing it.

    The caller must verify the original ComponentView/runtime/candidate owners;
    this conversion neither creates those grants nor admits a raw caller policy.
    Unsupported/environment/network options refuse before any service request.
    """
    switches = {"--die-with-parent", "--new-session", "--unshare-all", "--clearenv"}
    scopes = {"--proc": "/proc", "--dev": "/dev", "--tmpfs": "/tmp"}
    environment = {
        "HOME": "/tmp",
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": "/component-inputs/compiler",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    if not argv:
        raise StageGateError("container conversion needs the verified original tool policy")
    observed, seen_env, seen_scopes, mounts, cwd = set(), {}, {}, [], None
    index = 1
    try:
        while index < len(argv):
            flag = argv[index]
            if flag in switches:
                if flag in observed:
                    raise StageGateError("duplicate original isolation switch")
                observed.add(flag)
                index += 1
            elif flag == "--setenv":
                key, value = argv[index + 1 : index + 3]
                if key in seen_env:
                    raise StageGateError("duplicate original environment grant")
                seen_env[key] = value
                index += 3
            elif flag in scopes:
                if flag in seen_scopes:
                    raise StageGateError("duplicate original namespace grant")
                seen_scopes[flag] = argv[index + 1]
                index += 2
            elif flag in ("--ro-bind", "--bind"):
                source, destination = argv[index + 1 : index + 3]
                mounts.append(mount_from_source(Path(source), destination, read_only=flag == "--ro-bind"))
                index += 3
            elif flag == "--chdir":
                if cwd is not None:
                    raise StageGateError("duplicate original working directory")
                cwd = argv[index + 1]
                index += 2
            else:
                raise StageGateError("unsupported original tool policy option")
    except (IndexError, ValueError) as error:
        raise StageGateError("incomplete original tool policy grammar") from error
    if observed != switches or seen_env != environment or seen_scopes != scopes:
        raise StageGateError("original tool policy is missing its closed isolation/environment scopes")
    if cwd is None or cwd not in {row.destination for row in mounts}:
        raise StageGateError("original tool policy lacks its exact mounted working directory")
    return tuple(mounts)


def verify_owned_container(row, *, image_id, name, token):
    """Never inspect, signal or remove a foreign delegated command."""
    identity = row.get("Id", "")
    if (
        len(identity) != 64
        or any(char not in "0123456789abcdef" for char in identity)
        or row.get("Name") != "/" + name
        or row.get("Image") != image_id
        or row.get("Config", {}).get("Labels", {}).get("merlin.command-owner") != token
    ):
        raise StageGateError("service did not return this invocation's exact created container")
    return identity


def _verify_created_container(row, *, image_id, mounts, command, cwd, timeout_s, masks):
    config, host = row["Config"], row["HostConfig"]
    expected = {
        "NetworkMode": "none",
        "PidMode": "",
        "IpcMode": "private",
        "Privileged": False,
        "ReadonlyRootfs": True,
        "CapDrop": ["ALL"],
        "SecurityOpt": ["no-new-privileges=true"],
        "PidsLimit": 64,
        "Memory": 256 * 1024**2,
        "NanoCpus": 1000000000,
        "AutoRemove": False,
        "CgroupnsMode": "private",
    }
    if (
        row["Image"] != image_id
        or any(host.get(key) != value for key, value in expected.items())
        or any(host.get(key) for key in ("CapAdd", "Binds", "Devices", "DeviceRequests", "VolumesFrom"))
        or host.get("RestartPolicy", {}).get("Name") != "no"
        or config.get("User") != f"{os.getuid()}:{os.getgid()}"
        or config.get("WorkingDir") != cwd
        or config.get("Entrypoint") != ["/usr/bin/merlin-command-owner"]
        or config.get("Cmd") != [str(timeout_s), "--", *command]
        or host.get("Tmpfs")
        != {
            "/tmp": "rw,nosuid,nodev,noexec,size=64m",
            **{path: "ro,nosuid,nodev,noexec,size=16m" for path in masks},
        }
        or {(item["Source"], item["Destination"], item["RW"]) for item in row["Mounts"]}
        != {(str(item.source), item.destination, not item.read_only) for item in mounts}
    ):
        raise StageGateError("owned container differs from its exact original isolation/mount/command plan")
