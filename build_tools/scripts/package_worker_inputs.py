#!/usr/bin/env python3
"""Pack explicitly selected worker inputs; verify or extract them without discovery.

Requires installed Merlin core. Archives are private delivery artifacts, not compiler
certificates. Checksums establish byte identity, not authenticity: transfer the printed
archive SHA-256 through a trusted channel. No credentials or inputs are discovered.
Source trees and extraction parents must be controlled by the operator and not
concurrently mutated by another process running as the same user. This maintenance
tool is not an isolation boundary against an adversarial same-UID process.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import stat
import tarfile
from pathlib import Path, PurePosixPath

from merlin.common.artifacts import new_product
from merlin.common.digest import is_sha256
from merlin.common.jsonio import canonical_json, write_pretty_json
from merlin.common.paths import artifacts_dir

SCHEMA = "merlin.worker-input-delivery.v1"
BLOCK = 1024 * 1024


def _relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or "\\" in value or any(p in ("", ".", "..") for p in value.split("/")):
        raise ValueError(f"unsafe member path: {value!r}")
    return path


def _name(value: str) -> str:
    if (
        not value
        or value in (".", "..")
        or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for c in value)
    ):
        raise ValueError(f"input name must be one plain component: {value!r}")
    return value


def _stamp(path: Path) -> tuple:
    s = path.lstat()
    return (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def _real_directory(path: Path) -> None:
    """Refuse pre-existing symlinks anywhere in a lexical directory ancestry."""
    for ancestor in (*reversed(path.parents), path):
        if not stat.S_ISDIR(ancestor.lstat().st_mode):
            raise ValueError(f"directory ancestry must be real, not a symlink: {ancestor}")


def _file_digest(path: Path) -> str:
    before = _stamp(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        s = os.fstat(stream.fileno())
        if (s.st_dev, s.st_ino) != before[:2] or not stat.S_ISREG(s.st_mode):
            raise ValueError(f"source replaced: {path}")
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if _stamp(path) != before:
        raise ValueError(f"source changed while reading: {path}")
    return digest


def _inventory(inputs: dict[str, Path], *, stamps: dict | None = None) -> list[dict]:
    rows = [{"path": "inputs", "kind": "directory", "mode": 0o700}]
    for name, source in sorted(inputs.items()):
        root = source.absolute()
        _real_directory(root.parent)
        if root.is_symlink():
            raise ValueError(f"input root must not be a symlink: {root}")
        root = root.resolve(strict=True)
        boundary = root if root.is_dir() else root.parent

        def visit(path: Path, member: str) -> None:
            before = _stamp(path)
            mode = before[2]
            if stat.S_ISLNK(mode):
                link = os.readlink(path)
                if not link or PurePosixPath(link).is_absolute() or "\\" in link:
                    raise ValueError(f"only contained relative links are allowed: {path}")
                try:
                    resolved = path.resolve(strict=True)
                except (OSError, RuntimeError) as exc:
                    raise ValueError(f"broken or cyclic link: {path}") from exc
                if not resolved.is_relative_to(boundary):
                    raise ValueError(f"link escapes input root: {path}")
                rows.append({"path": member, "kind": "symlink", "mode": 0o777, "link": link})
            elif stat.S_ISDIR(mode):
                rows.append({"path": member, "kind": "directory", "mode": 0o700})
                for child in sorted(path.iterdir(), key=lambda p: p.name):
                    _relative(child.name)
                    visit(child, f"{member}/{child.name}")
            elif stat.S_ISREG(mode):
                rows.append(
                    {
                        "path": member,
                        "kind": "file",
                        "mode": 0o700 if mode & 0o111 else 0o600,
                        "size": before[3],
                        "sha256": _file_digest(path),
                    }
                )
            else:
                raise ValueError(f"unsupported source kind: {path}")
            if _stamp(path) != before:
                raise ValueError(f"source changed during inventory: {path}")
            if stamps is not None:
                stamps[str(path)] = before

        visit(root, f"inputs/{name}")
    return sorted(rows, key=lambda row: row["path"])


def _validate_manifest(manifest: dict) -> list[dict]:
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema", "inputs", "members", "provenance"}
        or manifest["schema"] != SCHEMA
    ):
        raise ValueError("unsupported delivery manifest")
    if (
        not isinstance(manifest["provenance"], dict)
        or not isinstance(manifest["inputs"], list)
        or not manifest["inputs"]
    ):
        raise ValueError("invalid provenance or input roster")
    names = manifest["inputs"]
    if any(not isinstance(n, str) or _name(n) != n for n in names) or names != sorted(set(names)):
        raise ValueError("invalid input names")
    rows = manifest["members"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("empty member roster")
    seen = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError("invalid member row")
        p = _relative(row["path"])
        if p.parts[0] != "inputs" or (len(p.parts) > 1 and p.parts[1] not in names) or str(p) in seen:
            raise ValueError("unexpected or duplicate member")
        kind = row.get("kind")
        fields = {"path", "kind", "mode"}
        fields |= {"size", "sha256"} if kind == "file" else {"link"} if kind == "symlink" else set()
        if set(row) != fields or type(row["mode"]) is not int:
            raise ValueError("invalid member fields")
        if kind == "directory" and row["mode"] == 0o700:
            pass
        elif (
            kind == "file"
            and row["mode"] in (0o600, 0o700)
            and type(row["size"]) is int
            and row["size"] >= 0
            and is_sha256(row["sha256"])
        ):
            pass
        elif (
            kind == "symlink"
            and row["mode"] == 0o777
            and isinstance(row["link"], str)
            and row["link"]
            and not PurePosixPath(row["link"]).is_absolute()
            and "\\" not in row["link"]
        ):
            pass
        else:
            raise ValueError("invalid member kind, permissions, size or digest")
        seen[str(p)] = row
    if [r["path"] for r in rows] != sorted(seen) or seen.get("inputs", {}).get("kind") != "directory":
        raise ValueError("noncanonical member order or missing root")
    for name in names:
        if f"inputs/{name}" not in seen:
            raise ValueError("missing selected input")
    edges = {key: [] for key in seen}
    for key, row in seen.items():
        parent = str(PurePosixPath(key).parent)
        if key != "inputs" and seen.get(parent, {}).get("kind") != "directory":
            raise ValueError("member parent is not a declared directory")
        if key != "inputs":
            edges[parent].append(key)
        if row["kind"] == "symlink":
            # Resolve through declared links only, without using the worker filesystem.
            pending = list(PurePosixPath(key).parent.parts) + row["link"].split("/")
            resolved, hops = [], 0
            while pending:
                part = pending.pop(0)
                if part in ("", "."):
                    continue
                if part == "..":
                    if len(resolved) <= 2:
                        raise ValueError("link escapes selected input")
                    resolved.pop()
                    continue
                resolved.append(part)
                node = seen.get("/".join(resolved))
                if node is None:
                    raise ValueError("link target is not a declared member")
                if node["kind"] == "symlink":
                    hops += 1
                    if hops > len(rows):
                        raise ValueError("cyclic link")
                    resolved.pop()
                    pending = node["link"].split("/") + pending
                elif pending and node["kind"] != "directory":
                    raise ValueError("link traverses a nondirectory")
            if resolved[:2] != list(PurePosixPath(key).parts[:2]):
                raise ValueError("link crosses selected inputs")
            edges[key].append("/".join(resolved))
    # Directory links must not introduce recursive traversal cycles either.
    colors = {}
    stack = [("inputs", False)]
    while stack:
        key, leaving = stack.pop()
        if leaving:
            colors[key] = 2
        elif colors.get(key) == 1:
            raise ValueError("cyclic directory link")
        elif colors.get(key) != 2:
            colors[key] = 1
            stack.append((key, True))
            stack.extend((child, False) for child in edges[key])
    return rows


def _header(row: dict) -> tarfile.TarInfo:
    info = tarfile.TarInfo(row["path"])
    info.mode = row["mode"]
    info.mtime = 0
    info.type = {"directory": tarfile.DIRTYPE, "file": tarfile.REGTYPE, "symlink": tarfile.SYMTYPE}[row["kind"]]
    if row["kind"] == "file":
        info.size = row["size"]
    elif row["kind"] == "symlink":
        info.linkname = row["link"]
    return info


def pack(inputs: dict[str, Path], *, target: str | None = None, provenance: dict | None = None) -> Path:
    """Produce a private, deterministic tar and inspectable sidecars under delivery/."""
    if not inputs:
        raise ValueError("at least one explicit input is required")
    if provenance is not None and not isinstance(provenance, dict):
        raise ValueError("provenance must be a JSON object")
    inputs = {_name(n): Path(p).absolute() for n, p in inputs.items()}
    if target is not None:
        _name(target)
    for path in inputs.values():
        if path.is_dir() and artifacts_dir().resolve().is_relative_to(path.resolve()):
            raise ValueError("delivery output cannot be inside an input")
    before = {}
    manifest = {
        "schema": SCHEMA,
        "inputs": sorted(inputs),
        "members": _inventory(inputs, stamps=before),
        "provenance": provenance or {},
    }
    rows = _validate_manifest(manifest)
    old_mask = os.umask(0o077)
    try:
        product = new_product(
            "delivery",
            target=target,
            version=1,
            update_latest=False,
            sources=[{"input": n, "path": str(p)} for n, p in sorted(inputs.items())],
            notes="Explicit private worker inputs; not a correctness certificate.",
        )
        product.path.chmod(0o700)
        archive = product.add_artifact("worker-inputs.tar")
        encoded = canonical_json(manifest) + b"\n"
        with archive.open("xb") as output, tarfile.open(fileobj=output, mode="w", format=tarfile.PAX_FORMAT) as bundle:
            info = tarfile.TarInfo("manifest.json")
            info.mode, info.size = 0o600, len(encoded)
            bundle.addfile(info, io.BytesIO(encoded))
            for row in rows:
                info = _header(row)
                if row["kind"] == "file":
                    parts = PurePosixPath(row["path"]).parts
                    source = inputs[parts[1]].joinpath(*parts[2:])
                    descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
                    with os.fdopen(descriptor, "rb") as stream:
                        bundle.addfile(info, stream)
                else:
                    bundle.addfile(info)
        after = {}
        if _inventory(inputs, stamps=after) != rows or after != before:
            raise ValueError("source inputs changed during packing; delivery is incomplete")
        report = verify(archive)
        write_pretty_json(product.add_artifact("inputs.json"), manifest)
        write_pretty_json(product.add_artifact("archive.json"), report)
        product.write_manifest()
        return product.path
    finally:
        os.umask(old_mask)


def verify(archive: Path, *, expected_sha256: str | None = None, extract: Path | None = None) -> dict:
    """Check every archive byte/member; optionally extract into a fresh private directory.

    Failed extraction leaves an incomplete private directory for inspection. It never
    overwrites an existing destination or follows pre-existing ancestor symlinks.
    The operator must prevent concurrent same-user mutation of source/parent trees.
    """
    archive = Path(archive)
    before = _stamp(archive)
    if not stat.S_ISREG(before[2]):
        raise ValueError("archive must be a regular file, not a symlink")
    if before[3] % tarfile.RECORDSIZE:
        raise ValueError("archive is truncated or contains trailing bytes")
    digest = _file_digest(archive)
    if expected_sha256 is not None and (not is_sha256(expected_sha256) or digest != expected_sha256):
        raise ValueError("archive SHA-256 differs from expected identity")
    destination = Path(extract).absolute() if extract is not None else None
    if destination is not None:
        _real_directory(destination.parent)
        if os.path.lexists(destination):
            raise ValueError("extraction destination must be new")
    links, count, byte_count = [], 0, 0
    with archive.open("rb") as source, tarfile.open(fileobj=source, mode="r|") as bundle:
        first = bundle.next()
        if (
            first is None
            or first.name != "manifest.json"
            or not first.isreg()
            or first.mode != 0o600
            or first.size > 32 * BLOCK
            or first.uid != 0
            or first.gid != 0
            or first.mtime != 0
            or first.uname
            or first.gname
        ):
            raise ValueError("missing or oversized private manifest")
        manifest_stream = bundle.extractfile(first)
        assert manifest_stream is not None
        encoded = manifest_stream.read()
        manifest = json.loads(encoded)
        if encoded != canonical_json(manifest) + b"\n":
            raise ValueError("manifest must use canonical JSON without duplicate keys")
        rows = _validate_manifest(manifest)
        if destination is not None:
            destination.mkdir(mode=0o700)
            destination.chmod(0o700)
        for row in rows:
            info = bundle.next()
            expected = _header(row)
            if (
                info is None
                or any(
                    getattr(info, key) != getattr(expected, key) for key in ("name", "type", "mode", "size", "linkname")
                )
                or info.uid != 0
                or info.gid != 0
                or info.mtime != 0
                or info.uname
                or info.gname
            ):
                raise ValueError(f"archive member differs from manifest: {row['path']}")
            target_path = destination.joinpath(*_relative(row["path"]).parts) if destination is not None else None
            if row["kind"] == "directory":
                if target_path is not None:
                    target_path.mkdir(mode=0o700)
                    target_path.chmod(0o700)
            elif row["kind"] == "symlink":
                links.append((target_path, row["link"]))
            else:
                incoming = bundle.extractfile(info)
                assert incoming is not None
                outgoing = (
                    os.fdopen(os.open(target_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb")
                    if target_path is not None
                    else None
                )
                if outgoing is not None:
                    os.fchmod(outgoing.fileno(), row["mode"])
                result, size = hashlib.sha256(), 0
                try:
                    while chunk := incoming.read(BLOCK):
                        result.update(chunk)
                        size += len(chunk)
                        if outgoing is not None:
                            outgoing.write(chunk)
                finally:
                    if outgoing is not None:
                        outgoing.close()
                if result.hexdigest() != row["sha256"] or size != row["size"]:
                    raise ValueError(f"payload digest or size mismatch: {row['path']}")
                byte_count += size
            count += 1
        payload_end = info.offset_data + ((info.size + tarfile.BLOCKSIZE - 1) // tarfile.BLOCKSIZE) * tarfile.BLOCKSIZE
        if before[3] < payload_end + 2 * tarfile.BLOCKSIZE:
            raise ValueError("archive is missing its termination blocks")
        if bundle.next() is not None:
            raise ValueError("unlisted archive member")
        while trailer := bundle.fileobj.read(BLOCK):
            if any(trailer):
                raise ValueError("unexpected bytes after archive terminator")
    if _stamp(archive) != before or _file_digest(archive) != digest:
        raise ValueError("archive changed during verification")
    if destination is not None:
        for path, link in links:
            assert path is not None
            path.symlink_to(link)
    return {
        "schema": SCHEMA,
        "archive_sha256": digest,
        "archive_size": before[3],
        "members": count,
        "payload_bytes": byte_count,
        "inputs": manifest["inputs"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    packing = commands.add_parser("pack", help="pack explicit NAME=PATH selections only")
    packing.add_argument("--input", action="append", required=True, metavar="NAME=PATH")
    packing.add_argument("--target")
    packing.add_argument("--provenance", type=Path, help="optional JSON object of role/revision metadata")
    checking = commands.add_parser("verify", help="verify; optionally extract to a new directory")
    checking.add_argument("archive", type=Path)
    checking.add_argument("--sha256", help="expected archive identity from a trusted channel")
    checking.add_argument("--extract", type=Path)
    args = parser.parse_args()
    if args.command == "pack":
        inputs = {}
        for item in args.input:
            name, separator, path = item.partition("=")
            if not separator or not path or name in inputs:
                parser.error("inputs must be unique NAME=PATH mappings")
            inputs[name] = Path(path)
        provenance = json.loads(args.provenance.read_text()) if args.provenance else None
        product = pack(inputs, target=args.target, provenance=provenance)
        print(product)
        print((product / "archive.json").read_text(), end="")
    else:
        print(json.dumps(verify(args.archive, expected_sha256=args.sha256, extract=args.extract), indent=2))


if __name__ == "__main__":
    main()
