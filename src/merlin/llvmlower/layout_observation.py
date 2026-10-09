"""Fresh selected LLVM layout observations through the public native API.

No width, stride, alignment or endian default is supplied here. The selected
LLVM library derives these from its actual DataLayout implementation. This is
an IR/tool observation, never physical allocation or target runtime authority.
"""

from __future__ import annotations

import json
import math
import shlex
import time
import weakref
from dataclasses import dataclass
from pathlib import Path

from merlin.common import invocation_record
from merlin.common.digest import sha256_file

_ISSUED = weakref.WeakKeyDictionary()
_SOURCE = r"""
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/Error.h>
#include <iostream>
#include <string>
int main() {
  std::string text;
  unsigned bits;
  if (!std::getline(std::cin, text) || !(std::cin >> bits) || bits == 0 ||
      bits > llvm::IntegerType::MAX_INT_BITS) return 1;
  auto layout = llvm::DataLayout::parse(text);
  if (!layout) { llvm::consumeError(layout.takeError()); return 2; }
  llvm::LLVMContext context;
  auto type = llvm::IntegerType::get(context, bits);
  std::cout << layout->getPointerSizeInBits(0) << " "
            << layout->getIndexSizeInBits(0) << " "
            << layout->getTypeAllocSize(type).getFixedValue() << " "
            << layout->getABITypeAlign(type).value() << " "
            << (layout->isLittleEndian() ? "little" : "big") << "\n";
}
"""


def _pin(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError("layout observation dependency is absent")
    return str(path), sha256_file(path)


def _deadline(timeout_s):
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not math.isfinite(timeout_s)
        or not 0 < timeout_s <= 600
    ):
        raise ValueError("layout observation requires an explicit finite bounded budget")
    return time.monotonic() + timeout_s


def _remaining(deadline):
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError("layout observation total budget exhausted")
    return left


def _run(argv, *, root, stage, deadline, inputs=(), outputs=(), dependencies=(), **kwargs):
    result = invocation_record.run(
        argv,
        directory=root,
        stage=stage,
        inputs=inputs,
        outputs=outputs,
        dependencies=(Path(__file__), *dependencies),
        capture_output=True,
        timeout=min(60, _remaining(deadline)),
        **kwargs,
    )
    if result.returncode:
        raise ValueError("selected native LLVM layout observation did not complete")
    return result.stdout


def _headers(path):
    text = path.read_text().replace("\\\n", " ")
    if ":" not in text:
        raise ValueError("selected native compiler omitted its header roster")
    return tuple(sorted({Path(token).resolve(strict=True) for token in shlex.split(text.split(":", 1)[1])}))


@dataclass(frozen=True, eq=False)
class LLVMLayoutObservation:
    """Live observation with its complete actual local invocation membership."""

    root: Path
    layout: str
    integer_bits: int
    pointer_bits: int
    index_bits: int
    allocation_stride: int
    abi_alignment: int
    byte_order: str
    source_pins: tuple[tuple[str, str], ...]
    records: tuple[tuple[str, str], ...]
    object_record: tuple[str, str] | None = None

    def verify(self):
        if _ISSUED.get(self) != self.record():
            raise ValueError("layout requires a fresh actual native observation")
        for path, sha in (*self.source_pins, *self.records):
            if _pin(path) != (path, sha):
                raise ValueError("layout observation dependency or invocation changed")
        actual = tuple(_pin(path) for path in sorted(self.root.rglob("invocation.json")))
        if actual != self.records:
            raise ValueError("layout observation lost an actual invocation")
        for path, _ in self.records:
            invocation_record.verify(Path(path))
        query_root = self.root / "native" if self.object_record is not None else self.root
        if (query_root / "query.txt").read_text() != f"{self.layout}\n{self.integer_bits}\n":
            raise ValueError("layout observation query changed")
        expected = (
            f"{self.pointer_bits} {self.index_bits} {self.allocation_stride} {self.abi_alignment} {self.byte_order}\n"
        )
        if (query_root / "result.txt").read_text() != expected:
            raise ValueError("layout observation result changed")
        return self.record()

    def record(self):
        return {
            "root": str(self.root),
            "layout": self.layout,
            "integer_bits": self.integer_bits,
            "pointer_bits": self.pointer_bits,
            "index_bits": self.index_bits,
            "allocation_stride": self.allocation_stride,
            "abi_alignment": self.abi_alignment,
            "byte_order": self.byte_order,
            "source_pins": self.source_pins,
            "records": self.records,
            "object_record": self.object_record,
            "scope": "selected public LLVM IR layout API; physical CPU/storage/runtime unproved",
        }


def observe_llvm_layout(*, layout, integer_bits, native_compiler, llvm_config, output_root, timeout_s=120):
    """Compile a fixed public LLVM accessor and execute the exact layout query.

    The coordinator owns independent selection of the tools. This mechanism
    observes their actual bytes, LLVM headers/libraries and processes; it does
    not authenticate their historical provenance or transitive system runtime.
    """
    deadline = _deadline(timeout_s)
    if type(layout) is not str or not layout or "\n" in layout or type(integer_bits) is not int or integer_bits <= 0:
        raise ValueError("layout observation needs an exact layout and positive original integer width")
    root = Path(output_root).absolute()
    if root.resolve() != root or root.exists():
        raise ValueError("layout observation needs a fresh direct output owner")
    compiler, config = (Path(path).resolve(strict=True) for path in (native_compiler, llvm_config))
    before = (_pin(compiler), _pin(config), _pin(__file__))
    root.mkdir(parents=True, mode=0o700)
    source, binary = root / "layout.cpp", root / "layout-probe"
    source.write_text(_SOURCE)
    query = root / "query.txt"
    query.write_text(f"{layout}\n{integer_bits}\n")
    flags = shlex.split(
        _run(
            [str(config), "--cxxflags", "--ldflags", "--libfiles", "core", "--system-libs"],
            root=root,
            stage="layout_native_flags",
            deadline=deadline,
            text=True,
        )
    )
    libraries = tuple(
        Path(flag).resolve(strict=True) for flag in flags if Path(flag).is_absolute() and Path(flag).is_file()
    )
    # Ask the actual compiler for every non-system header it consumes before
    # compiling the helper. The second invocation pins that complete roster.
    deps = root / "headers.d"
    _run(
        [str(compiler), *flags, "-MM", str(source), "-MF", str(deps)],
        root=root,
        stage="layout_native_headers",
        deadline=deadline,
        inputs=(source,),
        outputs=(deps,),
        dependencies=(config, *libraries),
    )
    headers = _headers(deps)
    actual_deps = root / "compiled-headers.d"
    _run(
        [str(compiler), str(source), *flags, "-MMD", "-MF", str(actual_deps), "-o", str(binary)],
        root=root,
        stage="layout_native_compile",
        deadline=deadline,
        inputs=(source,),
        outputs=(binary, actual_deps),
        dependencies=(config, *libraries, *headers),
    )
    if _headers(actual_deps) != headers:
        raise ValueError("native LLVM helper compilation changed its actual consumed header membership")
    with query.open("rb") as stream:
        output = _run(
            [str(binary)],
            root=root,
            stage="layout_native_query",
            deadline=deadline,
            inputs=(query,),
            dependencies=(*libraries, *headers),
            stdin=stream,
        )
    result = root / "result.txt"
    result.write_bytes(output)
    fields = output.decode().split()
    if len(fields) != 5 or any(not value.isdecimal() for value in fields[:4]) or fields[4] not in {"little", "big"}:
        raise ValueError("selected native LLVM accessor returned no closed layout observation")
    values = tuple(map(int, fields[:4]))
    if min(values) <= 0 or (_pin(compiler), _pin(config), _pin(__file__)) != before:
        raise ValueError("selected layout tool changed or returned invalid dimensions")
    pins = tuple(
        sorted(
            {
                _pin(path)
                for path in (compiler, config, source, binary, query, result, *libraries, *headers, Path(__file__))
            }
        )
    )
    records = tuple(_pin(path) for path in sorted(root.rglob("invocation.json")))
    observation = LLVMLayoutObservation(root, layout, integer_bits, *values, fields[4], pins, records)
    _ISSUED[observation] = observation.record()
    observation.verify()
    _remaining(deadline)
    (root / "layout_observation.json").write_text(json.dumps(observation.record(), indent=2) + "\n")
    return observation


def observe_compiled_layout(*, object_record, integer_bits, native_compiler, llvm_config, output_root, timeout_s=120):
    """Requery the actual ordinary object's compiler on its original LLVM input.

    Only the ordinary single-LLVM-input command suffix is supported. Revised
    transform objects need a separate preservation theorem and are not admitted
    here. The query shares every original driver option, replacing object output
    by LLVM emission; its data layout comes from the selected compiler's output.
    """
    from .target_data_layout import parse

    deadline = _deadline(timeout_s)
    record = Path(object_record).resolve(strict=True)
    document = invocation_record.verify(record)
    argv = document.get("argv", ())
    if (
        document.get("kind") != "subprocess"
        or document.get("stage") != "object"
        or len(argv) < 5
        or argv[-4] != "-c"
        or argv[-2] != "-o"
        or "-c" in argv[1:-4]
        or "-o" in argv[1:-4]
    ):
        raise ValueError("layout query requires the actual ordinary LLVM object command")
    original, obj = map(lambda path: Path(path).resolve(strict=True), (argv[-3], argv[-1]))
    if original.suffix != ".ll" or document["inputs"] != [{"path": str(original), "sha256": sha256_file(original)}]:
        raise ValueError("layout query has no exact original LLVM input")
    if {"path": str(obj), "sha256": sha256_file(obj)} not in document["outputs"]:
        raise ValueError("layout query has no exact actual object")
    compiler = Path(document["executable"]["path"])
    root = Path(output_root).absolute()
    if root.exists() or root.resolve() != root:
        raise ValueError("compiled layout query requires a fresh direct owner")
    root.mkdir(parents=True, mode=0o700)
    output = root / "selected.ll"
    dependencies = tuple(Path(row["path"]) for row in document["dependencies"])
    _run(
        [str(compiler), *argv[1:-4], "-S", "-emit-llvm", str(original), "-o", str(output)],
        root=root,
        stage="layout_selected_object_compiler",
        deadline=deadline,
        inputs=(original,),
        outputs=(output,),
        dependencies=(record, obj, *dependencies),
    )
    layout = parse(output.read_text())
    if not layout:
        raise ValueError("the actual selected object compiler emitted no data layout")
    observed = observe_llvm_layout(
        layout=layout,
        integer_bits=integer_bits,
        native_compiler=native_compiler,
        llvm_config=llvm_config,
        output_root=root / "native",
        timeout_s=_remaining(deadline),
    )
    invocation_record.verify(record)
    pins = tuple(
        sorted(
            {
                *observed.source_pins,
                _pin(record),
                _pin(compiler),
                _pin(original),
                _pin(obj),
                _pin(output),
                *(_pin(path) for path in dependencies),
            }
        )
    )
    records = tuple(_pin(path) for path in sorted(root.rglob("invocation.json")))
    combined = LLVMLayoutObservation(
        root,
        observed.layout,
        observed.integer_bits,
        observed.pointer_bits,
        observed.index_bits,
        observed.allocation_stride,
        observed.abi_alignment,
        observed.byte_order,
        pins,
        records,
        _pin(record),
    )
    _ISSUED[combined] = combined.record()
    combined.verify()
    _remaining(deadline)
    return combined
