"""Fail-closed inventory for a proposed fresh model2MLIR Python capture.

This tool never runs a loader, calls a capture, or upgrades an old receipt.
It names the external paths a sealed issuer must snapshot before it can run.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "merlin.python_capture_preflight.v1"
_CACHE_ENV = {"HF_HOME", "TORCH_HOME", "TRANSFORMERS_CACHE"}


def _loader_env_reads(source: str) -> set[str]:
    """Find literal os.environ.get/name subscripts without executing loader code."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    def is_environ(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and node.attr == "environ"
            and isinstance(node.value, ast.Name)
            and node.value.id == "os"
        )

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "get" and is_environ(node.func.value) and node.args:
                key = node.args[0]
            else:
                continue
        elif isinstance(node, ast.Subscript) and is_environ(node.value):
            key = node.slice
        else:
            continue
        if isinstance(key, ast.Constant) and isinstance(key.value, str) and key.value.isidentifier():
            names.add(key.value)
    return names


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path(value: str | Path, *, hash_file: bool = False) -> dict[str, Any]:
    path = Path(value).absolute()
    kind = "missing"
    if path.is_symlink():
        kind = "symlink"
    elif path.is_file():
        kind = "file"
    elif path.is_dir():
        kind = "directory"
    elif path.exists():
        kind = "other"
    return {
        "path": str(path),
        "kind": kind,
        "resolved": str(path.resolve(strict=False)),
        **({"sha256": _sha(path)} if kind == "file" and hash_file else {}),
    }


def _selected_path(value: str) -> dict[str, Any]:
    if not Path(value).is_absolute():
        return {"path": value, "kind": "unresolved_relative", "resolved": None}
    return _path(value, hash_file=True)


def _venv_home(venv: Path) -> Path | None:
    cfg = venv / "pyvenv.cfg"
    if not cfg.is_file():
        return None
    for line in cfg.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == "home":
            return Path(value.strip()).parent
    return None


def _editable_roots(site: Path) -> list[dict[str, Any]]:
    roots: dict[str, dict[str, Any]] = {}
    for pth in sorted(site.glob("*.pth")):
        lines = pth.read_text(errors="replace").splitlines()
        roots[f"pth:{pth.name}"] = {
            **_path(pth, hash_file=True),
            "startup_code": any(line.lstrip().startswith("import ") for line in lines),
        }
        for line in lines:
            line = line.strip()
            if line.startswith("/"):
                roots[f"pth-path:{pth.name}:{line}"] = _path(line)
    for finder in sorted(site.glob("__editable__*_finder.py")):
        roots[f"finder:{finder.name}"] = _path(finder, hash_file=True)
        try:
            tree = ast.parse(finder.read_text())
        except (UnicodeError, SyntaxError):
            continue
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            if not isinstance(target, ast.Name) or target.id not in {"MAPPING", "NAMESPACES"}:
                continue
            try:
                mapping = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            if isinstance(mapping, dict):
                for package, value in mapping.items():
                    for candidate in value if isinstance(value, list) else [value]:
                        if isinstance(candidate, str) and candidate.startswith("/"):
                            roots[f"editable:{package}:{candidate}"] = _path(candidate)
    return [dict(name=name, **value) for name, value in sorted(roots.items())]


def _elf_dependencies(binary: Path) -> dict[str, Any]:
    """Read ELF metadata without running the selected binary or its dynamic loader."""
    try:
        dynamic = subprocess.run(
            ["readelf", "-d", str(binary)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        )
        program = subprocess.run(
            ["readelf", "-l", str(binary)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"needed_sonames": [], "interpreter": None, "error": f"readelf unavailable: {type(exc).__name__}"}
    if dynamic.returncode or program.returncode:
        return {"needed_sonames": [], "interpreter": None, "error": "selected file is not inspectable ELF"}
    needed = sorted(
        {
            line.partition("[")[2].partition("]")[0]
            for line in dynamic.stdout.splitlines()
            if "(NEEDED)" in line and "[" in line and "]" in line
        }
    )
    interpreters = [
        line.partition("Requesting program interpreter:")[2].partition("]")[0].strip()
        for line in program.stdout.splitlines()
        if "Requesting program interpreter:" in line and "]" in line
    ]
    return {
        "needed_sonames": needed,
        "interpreter": _path(interpreters[0]) if interpreters else None,
        "error": None,
    }


def _receipt_audit(receipt: Path, loader: Path, m2m_root: Path) -> dict[str, Any]:
    """Compare *current* direct source bytes only; never attest a past execution."""
    result: dict[str, Any] = {
        "scope": "receipt_declared_direct_sources_only",
        "receipt": _path(receipt, hash_file=True),
        "status": "invalid_receipt",
        "historical_execution_verified": False,
        "source_closure_verified": False,
        "loader": None,
        "tool_sources": [],
        "errors": [],
    }
    if not receipt.is_file():
        result["errors"].append("Selected capture receipt is absent")
        return result
    try:
        payload = json.loads(receipt.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        result["errors"].append("Selected capture receipt is not readable JSON")
        return result
    if not isinstance(payload, dict) or payload.get("schema") != "m2m.capture-receipt.v1":
        result["errors"].append("Selected capture receipt has an unsupported schema")
        return result
    source = payload.get("source")
    tool = payload.get("tool")
    owner_hashes = tool.get("source_sha256") if isinstance(tool, dict) else None
    if not isinstance(source, dict) or not isinstance(owner_hashes, dict) or not owner_hashes:
        result["errors"].append("Receipt omits source or direct tool source hashes")
        return result
    selected_path = source.get("path")
    expected = source.get("sha256")
    if not isinstance(selected_path, str) or not Path(selected_path).is_absolute():
        result["errors"].append("Receipt loader path is not absolute")
        return result
    if Path(selected_path).absolute() != loader.absolute():
        result["errors"].append("Receipt loader path differs from the selected loader")
        return result
    if not isinstance(expected, str) or len(expected) != 64:
        result["errors"].append("Receipt loader digest is invalid")
        return result
    observed = _sha(loader) if loader.is_file() else None
    result["loader"] = {
        "expected_sha256": expected,
        "observed_sha256": observed,
        "status": "match" if expected == observed else "drift_or_missing",
    }
    for name, digest in sorted(owner_hashes.items()):
        if not isinstance(name, str) or not isinstance(digest, str):
            result["errors"].append("Receipt direct source entry is not a string pair")
            continue
        relative = PurePosixPath(name)
        if relative.is_absolute() or any(part in {".", ".."} for part in relative.parts) or "\\" in name:
            result["errors"].append("Receipt direct source path is unsafe")
            continue
        path = m2m_root.joinpath(*relative.parts)
        if not relative.parts or any(part.is_symlink() for part in (path, *path.parents) if part != Path("/")):
            result["errors"].append("Receipt direct source path is empty or traverses a symlink")
            continue
        observed = _sha(path) if path.is_file() else None
        result["tool_sources"].append(
            {
                "name": name,
                "expected_sha256": digest,
                "observed_sha256": observed,
                "status": "match" if observed == digest else "drift_or_missing",
            }
        )
    if result["errors"]:
        return result
    all_match = result["loader"]["status"] == "match" and all(
        row["status"] == "match" for row in result["tool_sources"]
    )
    result["status"] = "current_direct_sources_match" if all_match else "current_direct_sources_drift"
    return result


def inspect(
    *,
    worker: Path,
    loader: Path,
    m2m_root: Path,
    python: Path,
    selected_env: dict[str, str] | None = None,
    data_paths: list[Path] | None = None,
    capture_receipt: Path | None = None,
    required_env_names: list[str] | None = None,
) -> dict[str, Any]:
    """Read selected current state only; return no historical or sealed claim."""
    worker, loader, m2m_root, python = map(Path, (worker, loader, m2m_root, python))
    selected_env = dict(selected_env or {})
    data_paths = list(data_paths or [])
    required_env_names = list(required_env_names or [])
    if any(not name.isidentifier() for name in required_env_names):
        raise ValueError("Required environment names must be identifiers")
    manifest = loader.parent / "capture.toml"
    declared: dict[str, Any] = {}
    if manifest.is_file():
        declared = tomllib.loads(manifest.read_text())
        for key, value in (declared.get("env") or {}).items():
            selected_env.setdefault(str(key), str(value))
    venv = python.parent.parent
    site = venv / "lib" / f"python{declared.get('python', '3.12')}" / "site-packages"
    interpreter = python.resolve(strict=False)
    base = _venv_home(venv)
    inputs = {
        "worker": _path(worker, hash_file=True),
        "loader": _path(loader, hash_file=True),
        "loader_manifest": _path(manifest, hash_file=True),
        "m2m_root": _path(m2m_root),
        "m2m_package": _path(m2m_root / "m2m"),
        "venv": _path(venv),
        "venv_python": _path(python),
        "resolved_python": _path(interpreter, hash_file=True),
        "python_base": _path(base) if base else None,
        "site_packages": _path(site),
        "manifest_venv": _path(declared["venv"]) if isinstance(declared.get("venv"), str) else None,
    }
    editable = _editable_roots(site) if site.is_dir() else []
    loader_source = loader.read_text() if loader.is_file() else ""
    loader_env_names = _loader_env_reads(loader_source)
    env_names = sorted(loader_env_names | set(required_env_names) | (set(selected_env) & _CACHE_ENV))
    env_inputs = []
    for name in env_names:
        value = selected_env.get(name)
        path_like = name.endswith(("_NPZ", "_CKPT", "_CHECKPOINT", "_PATH", "_DIR", "_TOKEN_IDS")) or name in _CACHE_ENV
        env_inputs.append(
            {
                "name": name,
                "found_in_loader_source": name in loader_env_names,
                "declared_by_caller": name in required_env_names,
                "selection": "explicit" if value is not None else "not_selected",
                "path": _selected_path(value) if path_like and value else None,
            }
        )
    python_elf = (
        _elf_dependencies(interpreter)
        if interpreter.is_file()
        else {"needed_sonames": [], "interpreter": None, "error": "Python interpreter is absent"}
    )
    torch_candidates = sorted(site.glob("torch/_C*.so")) if site.is_dir() else []
    torch_elf = (
        _elf_dependencies(torch_candidates[0])
        if torch_candidates
        else {"needed_sonames": [], "interpreter": None, "error": "Torch extension is absent"}
    )
    required_paths = [
        *inputs.values(),
        *editable,
        *(row["path"] for row in env_inputs),
        python_elf["interpreter"],
        *(_path(path, hash_file=True) for path in data_paths),
    ]
    missing = sorted({item["path"] for item in required_paths if item and item["kind"] == "missing"})
    unresolved_relative = sorted(
        row["name"] for row in env_inputs if row["path"] and row["path"]["kind"] == "unresolved_relative"
    )
    blockers = [
        "No private immutable runtime/source/checkpoint snapshot was selected or executed.",
        "Python imports, native dlopen, and loader file reads are dynamic; "
        "this preflight is not a complete read closure.",
        "No empty-root, network-disabled bubblewrap capture and post-run byte verification occurred.",
        "The existing model2MLIR materialized receipt does not verify the capture process source closure.",
    ]
    if missing:
        blockers.append(f"Selected or referenced paths are absent: {missing}")
    unselected = [row["name"] for row in env_inputs if row["selection"] == "not_selected"]
    if unselected:
        blockers.append(
            f"Loader environment reads have no explicit selection: {unselected}; "
            "select their values or document that the chosen loader branch does not read them"
        )
    if unresolved_relative:
        blockers.append(f"Selected loader path values must be absolute: {unresolved_relative}")
    if inputs["manifest_venv"] and Path(inputs["manifest_venv"]["resolved"]) != venv.resolve():
        blockers.append("Selected interpreter does not match capture.toml's declared venv")
    if python_elf["error"] or torch_elf["error"]:
        blockers.append(
            f"Dynamic-library metadata inspection incomplete: "
            f"{[x for x in (python_elf['error'], torch_elf['error']) if x]}"
        )
    receipt_audit = _receipt_audit(capture_receipt, loader, m2m_root) if capture_receipt else None
    if receipt_audit:
        blockers.append(
            "A prior capture receipt can only compare its declared direct source hashes with current files; "
            "it cannot attest historical execution or complete Python source/runtime closure."
        )
        if receipt_audit["status"] != "current_direct_sources_match":
            blockers.append("Selected capture receipt's direct source bytes are absent, changed, or invalid")
    return {
        "schema": SCHEMA,
        "inspector_source_sha256": _sha(Path(__file__)),
        "status": "blocked_unsealed_python_capture",
        "source_closure_verified": False,
        "fresh_execution": False,
        "issued_capture": False,
        "inputs": inputs,
        "editable_and_startup_paths": editable,
        "loader_environment_reads": env_inputs,
        "caller_required_environment_names": sorted(set(required_env_names)),
        "environment_values_redacted": True,
        "unselected_loader_environment": unselected,
        "unselected_caller_required_environment": sorted(set(unselected) & set(required_env_names)),
        "explicit_data_paths": [_path(path, hash_file=True) for path in data_paths],
        "dynamic_libraries": {"python": python_elf, "torch_extension": torch_elf},
        "capture_receipt_audit": receipt_audit,
        "missing_paths": missing,
        "blockers": blockers,
        "next_step": (
            "On a filesystem with capacity, snapshot only the selected venv, resolved CPython base, editable import "
            "roots, worker/loader/M2M source, dependent OS/CUDA libraries, and exact checkpoint/data files. "
            "Then execute a new capture inside an empty-root bwrap namespace with network disabled, all inputs "
            "read-only, a fresh writable output/cache, cleared environment, and pre/post byte verification. "
            "A separately reviewed issuer/replay verifier is still required before Phase 0 admission."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("worker", "loader", "m2m_root", "python", "output"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--env", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--data-path", action="append", default=[], type=Path)
    parser.add_argument("--capture-receipt", type=Path)
    parser.add_argument("--require-env", action="append", default=[], metavar="NAME")
    args = parser.parse_args()
    selected_env = {}
    for item in args.env:
        name, sep, value = item.partition("=")
        if not sep or not name.isidentifier():
            parser.error("--env requires NAME=VALUE")
        selected_env[name] = value
    if any(not name.isidentifier() for name in args.require_env):
        parser.error("--require-env requires an environment variable name")
    result = inspect(
        worker=args.worker,
        loader=args.loader,
        m2m_root=args.m2m_root,
        python=args.python,
        selected_env=selected_env,
        data_paths=args.data_path,
        capture_receipt=args.capture_receipt,
        required_env_names=args.require_env,
    )
    output = args.output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": result["status"], "output": str(output), "missing_paths": result["missing_paths"]}))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
