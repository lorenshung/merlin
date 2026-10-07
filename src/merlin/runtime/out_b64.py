"""Strict full-value decoder for the opt-in bare-metal OUT_B64 v1 console ABI.

The wire format is printable text so existing simulator console capture remains
unchanged.  Each chunk carries actual little-endian container words, not a
digest, sample, or comparison with an expected result.  The normal numerical
checker still receives every reconstructed value.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass, field

RAW_CHUNK_CAP = 768
_MARKERS = frozenset({"OUT_B64_BEGIN", "OUT_B64_CHUNK", "OUT_B64_END"})


class OutB64Error(ValueError):
    """Malformed or incomplete lossless console output."""


def _decimal(value: str, *, label: str, positive: bool = False) -> int:
    if len(value) > 20 or not value.isascii() or not value.isdecimal() or str(int(value)) != value:
        raise OutB64Error(f"OUT_B64 {label} is not a canonical decimal integer")
    parsed = int(value)
    if positive and parsed <= 0:
        raise OutB64Error(f"OUT_B64 {label} must be positive")
    return parsed


def _hex(value: str, *, digits: int, label: str) -> int:
    if len(value) != digits or any(letter not in "0123456789abcdef" for letter in value):
        raise OutB64Error(f"OUT_B64 {label} is not canonical fixed-width hex")
    return int(value, 16)


@dataclass
class _Pending:
    name: str
    rows: int
    cols: int
    word_bytes: int
    signed: bool
    next_chunk: int = 0
    raw: bytearray = field(default_factory=bytearray)

    @property
    def expected_bytes(self) -> int:
        return self.rows * self.cols * self.word_bytes


class OutB64Decoder:
    """One sequential packet stream; unrelated legacy console lines pass through."""

    def __init__(self) -> None:
        self.pending: _Pending | None = None
        self.names: set[str] = set()

    def consume(self, parts: list[str], outputs: dict[str, list]) -> bool:
        """Consume one split line, returning whether it belongs to OUT_B64."""
        if not parts:
            return False
        marker = parts[0]
        if not marker.startswith("OUT_B64_"):
            if self.pending is not None:
                raise OutB64Error("OUT_B64 frame interrupted before END")
            return False
        if marker not in _MARKERS:
            raise OutB64Error("unknown OUT_B64 frame marker")
        if marker == "OUT_B64_BEGIN":
            if len(parts) != 7 or parts[1] != "v1" or self.pending is not None:
                raise OutB64Error("invalid or nested OUT_B64_BEGIN")
            name = parts[2]
            if not name or not name.isascii() or not all(c.isalnum() or c == "_" for c in name):
                raise OutB64Error("invalid OUT_B64 output name")
            if name in self.names or name in outputs:
                raise OutB64Error("duplicate OUT_B64 output name")
            rows = _decimal(parts[3], label="rows", positive=True)
            cols = _decimal(parts[4], label="cols", positive=True)
            width = _decimal(parts[5], label="word bytes", positive=True)
            if width not in (1, 2, 4, 8) or parts[6] not in ("s", "u"):
                raise OutB64Error("unsupported OUT_B64 container format")
            self.pending = _Pending(name, rows, cols, width, parts[6] == "s")
            self.names.add(name)
            return True
        pending = self.pending
        if pending is None:
            raise OutB64Error("OUT_B64 chunk or END without BEGIN")
        if marker == "OUT_B64_CHUNK":
            if len(parts) != 4:
                raise OutB64Error("malformed OUT_B64_CHUNK")
            sequence = _hex(parts[1], digits=8, label="sequence")
            length = _hex(parts[2], digits=4, label="length")
            if sequence != pending.next_chunk or length < 1 or length > RAW_CHUNK_CAP:
                raise OutB64Error("OUT_B64 chunk order or length differs")
            try:
                payload = base64.b64decode(parts[3], validate=True)
            except (binascii.Error, ValueError) as exc:
                raise OutB64Error("invalid OUT_B64 base64 payload") from exc
            if len(payload) != length or base64.b64encode(payload).decode("ascii") != parts[3]:
                raise OutB64Error("noncanonical or truncated OUT_B64 payload")
            if len(pending.raw) + length > pending.expected_bytes:
                raise OutB64Error("OUT_B64 payload exceeds declared output shape")
            pending.raw.extend(payload)
            pending.next_chunk += 1
            return True
        if len(parts) != 1 or len(pending.raw) != pending.expected_bytes:
            raise OutB64Error("OUT_B64_END before the complete declared output")
        words = [
            int.from_bytes(pending.raw[index : index + pending.word_bytes], "little", signed=pending.signed)
            for index in range(0, len(pending.raw), pending.word_bytes)
        ]
        outputs[pending.name] = [words[row * pending.cols : (row + 1) * pending.cols] for row in range(pending.rows)]
        self.pending = None
        return True

    def require_closed(self) -> None:
        if self.pending is not None:
            raise OutB64Error("OUT_B64 output ended without END")
