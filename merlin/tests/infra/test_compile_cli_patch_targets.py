"""A test that stands in for a compile helper must patch the module that DEFINES it.

``merlin.compile_cli`` re-exports its historically extracted helpers so callers keep resolving
``merlin.compile_cli.<name>``. A re-export is a second binding, though. ``run_matmul_on_mesh`` lives in
``merlin.compile.mesh`` and resolves ``_default_oot_package`` in THAT module's namespace, so
``monkeypatch.setattr(compile_cli, "_default_oot_package", fake)`` replaces a name nothing on that path
reads: the test keeps passing and no longer tests anything. The same holds between the modules of the
package (``mesh`` imports ``capacity``'s helpers by name).

This scans every test for the ways a module attribute gets replaced -- ``setattr(<alias>, "<name>", ...)``
(pytest's ``monkeypatch`` and ``MonkeyPatch()`` included), ``patch.object(<alias>, "<name>")``,
``setattr("merlin.compile_cli.<name>", ...)`` / ``patch("merlin.compile_cli.<name>")`` and plain
``<alias>.<name> = ...`` -- where the alias is bound to ``merlin.compile_cli`` or a ``merlin.compile``
module, and fails when the patched name is not defined in that module. A name it cannot resolve
statically (a variable that is not a loop over string literals) is a failure too: an unverifiable patch
is exactly how this hazard would come back.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

from merlin.common.paths import merlin_dir

_FACADE = "merlin.compile_cli"
_PACKAGE = "merlin.compile"
# Exact historical facade imports from the extraction commit. New module internals
# need not become facade exports, but these existing bindings must stay intact.
_FACADE_COMPAT_EXPORTS = {
    # These are the bindings imported by compile_cli when the facade was split.
    # Do not turn this into _defined_names(module): later helpers are not legacy API.
    "bundles": (
        "_IR_ELEMENT_ORDER", "_IR_ELEMENT_SPELLING", "_bundle_dir", "_capture_python",
        "ir_scalar_dtype",
    ),
    "capacity": (
        "_accumulator_capacity_elems", "_capacity_fit_tile", "_dtype_bits", "_dtype_bytes",
        "_operand_store_bytes", "_operand_store_capacity_elems", "capacity_fit",
        "declared_primitive_tile",
    ),
    "host_lane": (
        "_DTYPE_STRATEGY", "_verified_against_the_pinned_lane", "default_package",
        "host_lane_identity", "host_lane_pin_name",
    ),
    "mesh": (
        "_MESH_NTILE_WIDTH", "_certify_tile_via_executor", "_default_oot_package",
        "_matmul_via_bespoke_sim", "_matmul_via_oot_cert", "_matmul_via_program_oracle",
        "_mesh_rows", "_mesh_tile_binding", "_mesh_verify", "run_matmul_on_mesh",
        "tile_builder_op",
    ),
    "mesh_backend": (
        "_MESH_PKG_CACHE", "_MESH_PKG_LOCK", "_MESH_REFUSAL", "_MESH_RUN_SEQ",
        "_built_mesh_package", "_mesh_invocation_id", "_mesh_layer_id", "_refuse",
        "_requested_mesh_simulator", "_resolve_oot_mesh_simulator",
    ),
    "mesh_model": (
        "_int8_chain_policy", "_int8_chain_reference", "_int8_chain_step",
        "run_int8_chain_on_mesh", "run_whole_model_on_mesh",
    ),
    "mesh_reference": ("_accum_rel_tolerance", "_reference_on_datapath"),
    "model_preflight": ("preflight_model",),
}


def _is_compile_module(dotted: str | None) -> bool:
    return bool(dotted) and (dotted in (_FACADE, _PACKAGE) or dotted.startswith(_PACKAGE + "."))


def _compile_modules() -> list[str]:
    pkg = Path(inspect.getfile(importlib.import_module(_PACKAGE))).parent
    modules = {_FACADE}
    for path in pkg.rglob("*.py"):
        parts = path.relative_to(pkg).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        modules.add(".".join((_PACKAGE, *parts)))
    return sorted(modules)


def _defined_names(dotted: str) -> set[str]:
    """Top-level names a module binds by DEFINITION (def/class/assignment), never by import."""
    src = Path(inspect.getfile(importlib.import_module(dotted))).read_text(encoding="utf-8")
    out: set[str] = set()
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.Assign):
            out.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.add(node.target.id)
    return out


def _import_aliases(scope: ast.AST) -> dict[str, str]:
    """Names bound to a compile module by the import statements directly in ``scope`` (not nested defs)."""
    out: dict[str, str] = {}
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Import):
            for a in node.names:
                if _is_compile_module(a.name):
                    if a.asname:
                        out[a.asname] = a.name
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for a in node.names:
                dotted = f"{node.module}.{a.name}"
                if _is_compile_module(dotted):
                    out[a.asname or a.name] = dotted
        stack.extend(ast.iter_child_nodes(node))
    return out


def _module_of(expr: ast.AST, aliases: dict[str, str]) -> str | None:
    if isinstance(expr, ast.Name):
        return aliases.get(expr.id)
    # `import merlin.compile_cli` then `merlin.compile_cli.<name>`
    parts = []
    while isinstance(expr, ast.Attribute):
        parts.append(expr.attr)
        expr = expr.value
    if isinstance(expr, ast.Name):
        dotted = ".".join([expr.id, *reversed(parts)])
        return dotted if _is_compile_module(dotted) else None
    return None


def _name_values(expr: ast.AST, loops: dict[str, list[str]]) -> list[str] | None:
    """The attribute name(s) a patch targets, or None when they cannot be resolved statically."""
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return [expr.value]
    if isinstance(expr, ast.Name) and expr.id in loops:
        return loops[expr.id]
    return None


def _string_loops(scope: ast.AST) -> dict[str, list[str]]:
    """``for fn in ("a", "b"):`` -> {"fn": ["a", "b"]} (the only non-literal form the suite uses)."""
    out: dict[str, list[str]] = {}
    for node in ast.walk(scope):
        if (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and isinstance(node.iter, (ast.Tuple, ast.List))
            and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.iter.elts)
        ):
            out[node.target.id] = [e.value for e in node.iter.elts]
    return out


def _patches(path: Path):
    """Yield ``(lineno, module, [names] | None)`` for every compile-module attribute replaced in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    module_aliases = _import_aliases(tree)

    def visit(scope: ast.AST, aliases: dict[str, str]):
        loops = _string_loops(scope)
        for node in ast.iter_child_nodes(scope):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                yield from visit(node, {**aliases, **_import_aliases(node)})
                continue
            for sub in [node, *ast.walk(node)]:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and sub is not node:
                    continue
                if isinstance(sub, ast.Call):
                    fn = sub.func
                    fname = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
                    if fname not in ("setattr", "delattr", "object", "patch") or not sub.args:
                        continue
                    first = sub.args[0]
                    if isinstance(first, ast.Constant) and isinstance(first.value, str):
                        dotted, _, attr = first.value.rpartition(".")
                        if _is_compile_module(dotted):
                            yield sub.lineno, dotted, [attr]
                        continue
                    mod = _module_of(first, aliases)
                    if mod and fname != "patch":
                        yield sub.lineno, mod, (_name_values(sub.args[1], loops) if len(sub.args) > 1 else None)
                elif isinstance(sub, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                    targets = sub.targets if isinstance(sub, ast.Assign) else [sub.target]
                    for t in targets:
                        if isinstance(t, ast.Attribute):
                            mod = _module_of(t.value, aliases)
                            if mod:
                                yield sub.lineno, mod, [t.attr]

    # nested defs are visited through `visit`; de-duplicate what ast.walk re-reaches
    seen = set()
    for item in visit(tree, module_aliases):
        key = (item[0], item[1], tuple(item[2] or ()))
        if key not in seen:
            seen.add(key)
            yield item


def _test_files() -> list[Path]:
    return sorted((merlin_dir() / "tests").rglob("test_*.py"))


def test_every_compile_patch_targets_the_defining_module():
    defined = {m: _defined_names(m) for m in _compile_modules()}
    problems, n_patches = [], 0
    for path in _test_files():
        rel = path.relative_to(merlin_dir())
        for lineno, mod, names in _patches(path):
            if names is None:
                problems.append(f"{rel}:{lineno}: patches {mod} with a name this guard cannot resolve")
                continue
            for name in names:
                n_patches += 1
                if mod not in defined:
                    problems.append(f"{rel}:{lineno}: patches {mod}.{name}, not a module of the package")
                elif name not in defined[mod]:
                    home = [m for m, names_ in defined.items() if name in names_]
                    problems.append(
                        f"{rel}:{lineno}: patches {mod}.{name}, which {mod} does not define"
                        + (f" -- it is defined in {home[0]}; patch it there" if home else "")
                    )
    assert n_patches, "found no compile-module patches at all; the scanner has gone blind"
    assert not problems, (
        "a patch on a re-export does not reach the callers that resolve the name in its defining "
        "module, so the test stops testing anything:\n  " + "\n  ".join(problems)
    )


def test_the_scanner_catches_every_patch_form(tmp_path):
    """Mutation check on the guard itself: each form a re-export patch can take is reported."""
    probe = tmp_path / "test_probe.py"
    probe.write_text(
        "import merlin.compile_cli as cc\n"
        "from merlin import compile_cli\n"
        "import merlin.compile_cli\n"
        "def test_a(monkeypatch):\n"
        "    monkeypatch.setattr(cc, '_default_oot_package', None)\n"
        "    monkeypatch.setattr('merlin.compile_cli._mesh_tile_binding', None)\n"
        "    cc._operand_store_bytes = None\n"
        "    merlin.compile_cli._matmul_via_oot_cert = None\n"
        "    for fn in ('host_lane_pin_name',):\n"
        "        monkeypatch.setattr(compile_cli, fn, None)\n"
        "def test_b(monkeypatch):\n"
        "    from merlin.compile import mesh as M\n"
        "    monkeypatch.setattr(M, 'capacity_fit', None)\n"
        "    monkeypatch.setattr(M, '_default_oot_package', None)\n"
        "    from merlin.compile import scheduling as S\n"
        "    monkeypatch.setattr(S, 'BlockSchedule', None)\n",
        encoding="utf-8",
    )
    found = [(mod, n) for _, mod, names in _patches(probe) for n in (names or [])]
    assert ("merlin.compile_cli", "_default_oot_package") in found
    assert ("merlin.compile_cli", "_mesh_tile_binding") in found
    assert ("merlin.compile_cli", "_operand_store_bytes") in found
    assert ("merlin.compile_cli", "_matmul_via_oot_cert") in found
    assert ("merlin.compile_cli", "host_lane_pin_name") in found
    assert ("merlin.compile.mesh", "capacity_fit") in found
    assert ("merlin.compile.scheduling", "BlockSchedule") in found
    defined_cli = _defined_names("merlin.compile_cli")
    defined_mesh = _defined_names("merlin.compile.mesh")
    assert "_default_oot_package" not in defined_cli, "re-exported, so it must be reported"
    assert "capacity_fit" not in defined_mesh, "imported into mesh from capacity, so it must be reported"
    assert "_default_oot_package" in defined_mesh, "defined in mesh, so patching it there is right"
    assert _PACKAGE in _compile_modules()
    assert "merlin.compile.scheduling" in _compile_modules()
    assert "merlin.compile.scheduling.block_schedule" in _compile_modules()
    assert "BlockSchedule" not in _defined_names("merlin.compile.scheduling")


def test_the_facade_still_resolves_its_historical_compatibility_names():
    """Keep legacy facade bindings without exporting every new module's private helpers."""
    cli = importlib.import_module(_FACADE)
    absent = object()
    for suffix, names in _FACADE_COMPAT_EXPORTS.items():
        mod = f"{_PACKAGE}.{suffix}"
        m = importlib.import_module(mod)
        defined = _defined_names(mod)
        for name in names:
            assert name in defined, f"historical {mod}.{name} is no longer defined there"
            assert getattr(cli, name, absent) is getattr(m, name), f"{_FACADE}.{name} is not {mod}.{name}"
