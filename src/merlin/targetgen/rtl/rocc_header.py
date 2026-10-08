"""Derive RoCC register usage by preprocessing selected C header call forms."""

from __future__ import annotations

import ast
import hashlib
import shlex
import subprocess
from pathlib import Path


def _arguments(text: str, start: int) -> tuple[list[str], int]:
    depth, begin = 1, start
    args = []
    for i in range(start, len(text)):
        char = text[i]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return [*args, text[begin:i].strip()], i + 1
        elif char == "," and depth == 1:
            args.append(text[begin:i].strip())
            begin = i + 1
    raise ValueError("unterminated RoCC macro call")


def _calls(text: str, prefix: str) -> list[tuple[str, list[str]]]:
    # Consume logical preprocessor lines, retaining macro bodies but excluding
    # their formal parameter lists. This reader accepts identifier funct operands
    # only; quoted payloads and computed funct expressions fail closed.
    constants = set()
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[0] == "#define":
            try:
                int(fields[2], 0)
            except ValueError:
                continue
            constants.add(fields[1])
    calls = []
    for line in text.replace("\\\n", " ").splitlines():
        line = line.split("//", 1)[0]
        if line.lstrip().startswith("#define "):
            head = line.lstrip()[len("#define ") :]
            if "(" in head and " " not in head.partition("(")[0]:
                _, end = _arguments(head, head.index("(") + 1)
                line = head[end:]
        pos = 0
        while (pos := line.find(prefix, pos)) >= 0:
            end = pos
            while end < len(line) and (line[end].isalnum() or line[end] == "_"):
                end += 1
            name = line[pos:end]
            rest = end
            while rest < len(line) and line[rest].isspace():
                rest += 1
            if rest == len(line) or line[rest] != "(":
                pos = end
                continue
            args, pos = _arguments(line, rest + 1)
            if args[-1] in constants and args[0].isidentifier():
                calls.append((name, args))
    return calls


def decode_expansion(expansion: str) -> dict:
    """Read the actual inline asm template, including its register operands."""
    _, found, body = expansion.partition("asm volatile(")
    if not found:
        # Preprocessors may retain whitespace between volatile and its paren.
        _, found, body = expansion.partition("asm volatile (")
    if not found:
        raise ValueError("RoCC expansion has no inline asm")
    template = body.partition(":")[0].strip()
    literals = shlex.split(template, posix=False)
    instruction = "".join(ast.literal_eval(token) for token in literals)
    if not instruction.startswith(".insn r "):
        raise ValueError("RoCC expansion has no .insn r template")
    fields = [field.strip() for field in instruction[len(".insn r ") :].split(",")]
    if len(fields) != 6:
        raise ValueError("RoCC instruction does not have six fields")
    slot, funct3, funct = fields[:3]
    usage = {name: operand != "x0" for name, operand in zip(("rd", "rs1", "rs2"), fields[3:], strict=True)}
    if any(operand != "x0" and not operand.startswith("%") for operand in fields[3:]):
        raise ValueError("unsupported inline asm register operand")
    # RoCC's RISC-V funct3 packs xd/xs1/xs2 in that order (bits 2/1/0).
    encoded = sum(int(usage[name]) << bit for name, bit in (("rd", 2), ("rs1", 1), ("rs2", 0)))
    if int(funct3, 0) != encoded:
        raise ValueError("RoCC register operands disagree with funct3 flags")
    return {
        "custom_slot": int(slot.removeprefix("CUSTOM_")),
        "funct": int(funct, 0),
        "funct3": encoded,
        "registers": usage,
        "asm": instruction,
    }


def extract(header: Path, *, include_root: Path, prefix: str, compiler: str = "cc") -> dict:
    """Expand each observed call form without compiling or executing device code.

    Operand payloads become dummy C variables. The selected slot and funct tokens
    remain unchanged and are resolved by the real C preprocessor and headers.
    Unsupported forms refuse; absent header calls remain unknown to the caller.
    """
    calls = _calls(header.read_text(), prefix)
    if not calls:
        raise ValueError("selected header has no RoCC call forms")
    probes = []
    for index, (name, args) in enumerate(calls):
        operands = [args[0], *[f"probe_operand_{i}" for i in range(len(args) - 2)], args[-1]]
        probes.append(f"ROCC_PROBE_BEGIN_{index}\n{name}({', '.join(operands)})\nROCC_PROBE_END_{index}")
    source = f'#include "{header.resolve()}"\n' + "\n".join(probes)
    command = [compiler, "-E", "-x", "c", "-I", str(include_root), "-"]
    result = subprocess.run(command, input=source, text=True, capture_output=True, check=True)
    paths = {header.resolve(), Path(__file__).resolve()}
    for line in result.stdout.splitlines():
        if line.startswith("# ") and '"' in line:
            path = Path(line.split('"')[1])
            if path.is_absolute() and path.is_file():
                paths.add(path)
    rows = {}
    for index, (name, _) in enumerate(calls):
        expansion = result.stdout.partition(f"ROCC_PROBE_BEGIN_{index}\n")[2].partition(f"ROCC_PROBE_END_{index}")[0]
        row = decode_expansion(expansion)
        key = str(row.pop("funct"))
        row["call_macro"] = name
        if key in rows and rows[key] != row:
            raise ValueError(f"conflicting header usage for funct {key}")
        rows[key] = row
    return {
        "method": "selected C call forms expanded by compiler; .insn flags cross-checked against operands",
        "by_funct": rows,
        "command": command,
        "preprocessed_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
        "compiler_version": subprocess.run([compiler, "--version"], text=True, capture_output=True, check=True).stdout,
        "sources": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(paths)],
    }
