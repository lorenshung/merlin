"""Stage renderer-supplied constant operands beside a runner-owned C harness.

The target renderer owns tensor layout and bytes. The runner owns filenames,
assembly, and linking so no target-specific source is needed in the core.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Mapping

_SYMBOL = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")


def render_blob_asm(symbol: str, payload_name: str, *, align: int, elems: int) -> str:
    """Define an aligned C-visible symbol from exact binary input bytes."""
    if not isinstance(symbol, str) or not _SYMBOL.fullmatch(symbol):
        raise ValueError(f"invalid harness blob symbol {symbol!r}")
    if (not isinstance(payload_name, str) or Path(payload_name).name != payload_name
            or not payload_name.endswith(".bin")):
        raise ValueError(f"invalid harness blob payload name {payload_name!r}")
    if (type(align) is not int or type(elems) is not int or
            align < 1 or align & (align - 1) or elems < 1):
        raise ValueError("harness blob alignment must be a power of two and element count positive")
    return (
        "  .section .rodata\n"
        f"  .balign {align}\n"
        f"  .globl {symbol}\n"
        f"{symbol}:\n"
        f'  .incbin "{payload_name}"\n'
        f"  .size {symbol}, . - {symbol}\n"
        f"  /* {elems} elements */\n"
    )


def stage_harness_blobs(workdir: Path, blobs: Mapping[str, Mapping]) -> tuple[Path, ...]:
    """Validate and materialize a renderer's sidecars for the current build."""
    sources = []
    records = []
    for symbol, spec in sorted(blobs.items()):
        if not isinstance(spec, Mapping) or set(spec) != {"bytes", "align", "elems"}:
            raise ValueError(f"invalid harness blob declaration for {symbol!r}")
        payload = spec["bytes"]
        align, elems = spec["align"], spec["elems"]
        if (not isinstance(payload, bytes) or not payload or
                type(align) is not int or type(elems) is not int or
                elems < 1 or len(payload) % elems):
            raise ValueError(f"invalid harness blob bytes/extent for {symbol!r}")
        name = f"harness_blob_{symbol}.bin"
        asm = render_blob_asm(symbol, name, align=align, elems=elems)
        blob_path = workdir / name
        source = workdir / f"harness_blob_{symbol}.S"
        if blob_path.is_symlink() or source.is_symlink():
            raise ValueError("harness blob output path is a symlink")
        blob_path.write_bytes(payload)
        source.write_text(asm, encoding="utf-8")
        sources.append(source)
        records.append({
            "symbol": symbol,
            "file": name,
            "assembly": source.name,
            "object": source.with_suffix(".o").name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "alignment": align,
            "elements": elems,
        })
    receipt = workdir / "harness_blobs.json"
    if receipt.is_symlink():
        raise ValueError("harness blob receipt path is a symlink")
    if records:
        receipt.write_text(json.dumps({"schema": "merlin.harness_blobs.v1", "blobs": records},
                                      sort_keys=True, indent=2) + "\n", encoding="utf-8")
    else:
        receipt.unlink(missing_ok=True)
    return tuple(sources)
