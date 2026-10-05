"""What a whole-model measurement is a function of: the package, the program, the builder's whole
static module closure, the machine -- and therefore which store answers a request."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from merlin_experiments.phase2.whole_model_measured import identity as I


def _toy_package(root: Path) -> Path:
    pkg = root / "toybuild"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "builder.py").write_text(
        textwrap.dedent(
            """
            def build(package_dir, *, target, out_dir, **options):
                from .lowering import lower  # a lazy, relative import: part of what the builder runs
                return lower()
            """
        )
    )
    (pkg / "lowering.py").write_text("from toybuild import helpers\n\ndef lower():\n    return helpers.VALUE\n")
    (pkg / "helpers.py").write_text("VALUE = 1\n")
    (pkg / "unrelated.py").write_text("NOT_IMPORTED = 1\n")
    core = root / "toycore"
    core.mkdir()
    (core / "__init__.py").write_text("")
    (core / "passes.py").write_text("PASS = 1\n")
    (pkg / "helpers.py").write_text("from toycore.passes import PASS\n\nVALUE = PASS\n")
    return pkg


@pytest.fixture
def toy(tmp_path, monkeypatch):
    pkg = _toy_package(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in [m for m in sys.modules if m.partition(".")[0] in ("toybuild", "toycore")]:
        monkeypatch.delitem(sys.modules, name)
    return pkg


def test_the_closure_follows_lazy_and_relative_imports_but_not_unimported_modules(toy):
    closure = I.module_closure("toybuild.builder")
    assert set(closure) == {
        "toybuild/__init__.py",
        "toybuild/builder.py",
        "toybuild/lowering.py",
        "toybuild/helpers.py",
        "toycore/__init__.py",
        "toycore/passes.py",
    }
    assert not any(key.startswith(("json", "pathlib")) for key in closure)  # the interpreter's own


def test_a_change_two_imports_away_moves_the_store(toy, tmp_path):
    """The known defect: the store key hashed only the builder's spec STRING, so a behaviour-changing fix
    in a module the builder calls into (here two imports away) served the old builder's answers."""
    machine = {"kind": "spike", "target": "toy"}
    builder = {"spec": "toybuild.builder:build", "sha256": None}
    before = I.store_root_for(tmp_path / "store", builder=builder, machine=machine)
    (toy / "helpers.py").write_text("from toycore.passes import PASS\n\nVALUE = PASS + 1\n")
    after = I.store_root_for(tmp_path / "store", builder=builder, machine=machine)
    assert before != after
    (toy / "unrelated.py").write_text("NOT_IMPORTED = 2\n")
    assert I.store_root_for(tmp_path / "store", builder=builder, machine=machine) == after
    # A fix in ANOTHER source package the builder calls into (the core) moves the store too.
    (tmp_path / "toycore" / "passes.py").write_text("PASS = 2\n")
    assert I.store_root_for(tmp_path / "store", builder=builder, machine=machine) != after


def test_the_identity_does_not_depend_on_what_this_process_imported(toy, tmp_path):
    """A digest of the RESIDENT modules gave the launcher and a fresh worker different identities for
    the same bytes; a static closure is a function of the files alone."""
    here = I.builder_identity("toybuild.builder:build")["sha256"]
    import toybuild.builder  # noqa: F401
    import toybuild.unrelated  # noqa: F401

    assert I.builder_identity("toybuild.builder:build")["sha256"] == here
    code = (
        f"import sys; sys.path.insert(0, {str(tmp_path)!r});"
        "from merlin_experiments.phase2.whole_model_measured import identity as I;"
        "print(I.builder_identity('toybuild.builder:build')['sha256'])"
    )
    fresh = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert fresh == here


def test_an_unlocatable_builder_module_is_refused_not_narrowed(tmp_path):
    with pytest.raises(I.IdentityError):
        I.builder_identity("no_such_package_anywhere.builder:build")


def test_a_file_builder_is_admitted_only_at_its_pin(tmp_path):
    path = tmp_path / "b.py"
    path.write_text("def build(*a, **k):\n    return {}\n")
    with pytest.raises(I.IdentityError):
        I.load_builder(f"{path}:build", expected_sha256="0" * 64)
    assert callable(I.load_builder(f"{path}:build", expected_sha256=I.sha256_file(path)))
    with pytest.raises(I.IdentityError):
        I.builder_identity(f"{path}:build")


def test_documentation_edits_do_not_change_the_program_digest(tmp_path):
    pkg = tmp_path / "p"
    (pkg / "docs").mkdir(parents=True)
    (pkg / "manifest.yaml").write_text("components:\n  lowering: [lowering/]\n")
    (pkg / "lowering").mkdir()
    (pkg / "lowering" / "conv.py").write_text("X = 1\n")
    (pkg / "docs" / "notes.md").write_text("a")
    first = I.program_digest(pkg), I.package_digest(pkg)
    (pkg / "docs" / "notes.md").write_text("b")
    assert I.program_digest(pkg) == first[0] and I.package_digest(pkg) != first[1]
    (pkg / "lowering" / "conv.py").write_text("X = 2\n")
    assert I.program_digest(pkg) != first[0]
