import copy
import hashlib
import json
from pathlib import Path

import pytest

from merlin.runtime.numeric_provider_identity import validate_numeric_provider_identity


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def seal(tmp_path):
    root = tmp_path / "with spaces"
    root.mkdir()
    source, compiler, library, deps = [root / name for name in ("provider.c", "compiler", "provider.so", "provider.d")]
    for path in (source, compiler, library):
        path.write_bytes(path.name.encode())
    escape = lambda path: str(path).replace(" ", "\\ ")
    deps.write_text(escape(library) + ": \\\n " + escape(source) + "\n")
    cmd = [str(compiler), "-shared", str(source), "-o", str(library), "-MD", "-MF", str(deps)]
    manifest = {
        "compile": cmd,
        "compile_commands": [cmd],
        "compile_cwd": str(root),
        "compiler_sha256": sha(compiler),
        "dependency_file_sha256": sha(deps),
        "transitive_compile_dependencies": {str(source): sha(source)},
        "local_pins": {"provider.c": sha(source), "provider.so": sha(library)},
    }
    path = root / "manifest.json"
    witness = {
        "workspace_bytes": 128,
        "workspace_alignment": 8,
        "native_shared_sha256": sha(library),
        "numeric_dependency_pins": {str(p): sha(p) for p in (source, compiler, library, deps)},
    }

    def check(mutation=None, query_size=128, query_alignment=8):
        m, w = copy.deepcopy(manifest), copy.deepcopy(witness)
        if mutation:
            mutation(m, w)
        path.write_text(json.dumps(m))
        w.setdefault("complete_compile_manifest_sha256", sha(path))
        return validate_numeric_provider_identity(
            w,
            manifest_path=path,
            native_library_path=library,
            queried_workspace_bytes=query_size,
            queried_workspace_alignment=query_alignment,
        )

    return check


def test_exact_identity_and_space_paths(seal):
    assert seal()["validated_dependency_count"] == 4


@pytest.mark.parametrize(
    "mutation",
    [
        lambda m, w: w.update(workspace_bytes=64),
        lambda m, w: w.update(workspace_alignment=4),
        lambda m, w: w.update(workspace_bytes=True),
        lambda m, w: w.update(native_shared_sha256="0" * 64),
        lambda m, w: w.update(complete_compile_manifest_sha256="0" * 64),
        lambda m, w: m.update(compile_commands=[["old-compiler"]]),
        lambda m, w: m.update(transitive_compile_dependencies={}),
        lambda m, w: m.update(dependency_file_sha256="0" * 64),
        lambda m, w: m.update(compiler_sha256="0" * 64),
        lambda m, w: m.update(compile_cwd="relative"),
        lambda m, w: w.update(numeric_dependency_pins={}),
        lambda m, w: m["local_pins"].update({"provider.c": "0" * 64}),
        lambda m, w: m["local_pins"].update({"provider.so": "0" * 64}),
        lambda m, w: m["local_pins"].update({"compiler": "0" * 64}),
        lambda m, w: m["local_pins"].update({"provider.d": "0" * 64}),
        lambda m, w: m["local_pins"].update({"./provider.so": "0" * 64}),
    ],
)
def test_stale_or_conflicting_authoritative_fields_refused(seal, mutation):
    with pytest.raises(ValueError):
        seal(mutation)


@pytest.mark.parametrize("size,alignment", [(True, 8), (0, 8), (128, 3), (128, False)])
def test_query_identity_refused(seal, size, alignment):
    with pytest.raises(ValueError):
        seal(query_size=size, query_alignment=alignment)


def test_stronger_declared_alignment_is_safe(seal):
    result = seal(lambda m, w: w.update(workspace_alignment=64))
    assert result["workspace_alignment"] == 64
    assert result["queried_minimum_alignment"] == 8
