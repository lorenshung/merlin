"""Cross-check a numerical provider's artifact and queried workspace identity.

This validates an evidence seal, not mathematical correctness. Callers still
supply the numerical proof and execute the provider's workspace query under
its admitted ABI. Historical manifests must live outside authoritative fields. A declared power-of-two
alignment may exceed the actual minimum alignment returned by the provider.
"""

import hashlib
import json
import shlex
from collections.abc import Mapping
from pathlib import Path


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _positive_integer(value):
    return type(value) is int and value > 0


def validate_numeric_provider_identity(
    witness, *, manifest_path, native_library_path, queried_workspace_bytes, queried_workspace_alignment
):
    """Refuse conflicting current workspace, native image, recipe or file pins.

    The manifest records one actual native compile command, its cwd, compiler
    pin, a compiler-produced dependency file, and the complete dependency map.
    The caller's byte/alignment values must come from the selected native image's
    actual ABI queries. No library is loaded or executed by this validator.
    """
    if not isinstance(witness, Mapping):
        raise ValueError("numerical witness must be a mapping")
    size, alignment = queried_workspace_bytes, queried_workspace_alignment
    if not _positive_integer(size) or not _positive_integer(alignment) or alignment & (alignment - 1):
        raise ValueError("queried workspace identity is invalid")
    if (
        type(witness.get("workspace_bytes")) is not int
        or witness["workspace_bytes"] != size
        or type(witness.get("workspace_alignment")) is not int
        or witness["workspace_alignment"] < alignment
        or witness["workspace_alignment"] & (witness["workspace_alignment"] - 1)
    ):
        raise ValueError("witness conflicts with queried workspace identity")
    manifest_path = Path(manifest_path).resolve()
    library = Path(native_library_path).resolve()
    if witness.get("native_shared_sha256") != _sha(library):
        raise ValueError("witness native library identity mismatch")
    if witness.get("complete_compile_manifest_sha256") != _sha(manifest_path):
        raise ValueError("witness compile manifest identity mismatch")
    manifest = json.loads(manifest_path.read_text())
    command = manifest.get("compile")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(arg, str) for arg in command)
        or manifest.get("compile_commands") != [command]
    ):
        raise ValueError("conflicting native compile command identities")
    cwd = Path(manifest.get("compile_cwd", ""))
    if not cwd.is_absolute():
        raise ValueError("explicit absolute compile cwd required")

    def resolve(value):
        path = Path(value)
        return (path if path.is_absolute() else cwd / path).resolve()

    def argument(flag):
        if command.count(flag) != 1 or command.index(flag) + 1 == len(command):
            raise ValueError("unique compile output and dependency arguments required")
        return resolve(command[command.index(flag) + 1])

    if argument("-o") != library:
        raise ValueError("compile command does not produce selected native library")
    compiler = resolve(command[0])
    if _sha(compiler) != manifest.get("compiler_sha256"):
        raise ValueError("compiler executable identity mismatch")
    depfile = argument("-MF")
    if _sha(depfile) != manifest.get("dependency_file_sha256"):
        raise ValueError("compiler dependency file identity mismatch")
    text = depfile.read_text().replace("\\\n", " ")
    target, separator, dependencies = text.partition(":")
    if not separator or [resolve(x) for x in shlex.split(target)] != [library]:
        raise ValueError("dependency file output identity mismatch")
    actual_paths = {str(resolve(x)) for x in shlex.split(dependencies)}
    declared = manifest.get("transitive_compile_dependencies")
    if not isinstance(declared, dict) or set(declared) != actual_paths:
        raise ValueError("incomplete or stale transitive dependency identities")
    local = manifest.get("local_pins")
    if not isinstance(local, dict) or not local:
        raise ValueError("explicit native local artifact pins required")
    current = {}

    def add(path, digest):
        if path in current and current[path] != digest:
            raise ValueError("conflicting authoritative dependency identities: " + path)
        current[path] = digest

    for name, digest in local.items():
        add(str((manifest_path.parent / name).resolve()), digest)
    for path, digest in declared.items():
        add(path, digest)
    add(str(compiler), manifest["compiler_sha256"])
    add(str(depfile), manifest["dependency_file_sha256"])
    add(str(library), witness["native_shared_sha256"])
    witness_pins = witness.get("numeric_dependency_pins")
    if not isinstance(witness_pins, dict):
        raise ValueError("explicit numerical dependency pins required")
    for path, digest in current.items():
        if _sha(path) != digest or witness_pins.get(path) != digest:
            raise ValueError("numerical dependency identity mismatch: " + path)
    return {
        "workspace_bytes": size,
        "queried_minimum_alignment": alignment,
        "workspace_alignment": witness["workspace_alignment"],
        "native_shared_sha256": _sha(library),
        "complete_compile_manifest_sha256": _sha(manifest_path),
        "validated_dependency_count": len(current),
    }
