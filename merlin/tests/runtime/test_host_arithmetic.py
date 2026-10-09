"""Explicit shared host selection and actual native arithmetic emission."""

import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.llvmlower.source_fma_batch import SourceFmaBatchContract
from merlin.runtime.host_arithmetic import SourceFmaBatchCapability, SourceFmaPairCapability
from merlin.runtime.host_outward import emit_fixed_outward_f64_header

PAIR = SourceFmaPairCapability("rv64gc", "lp64d", True, True, True, True)
BATCH = SourceFmaBatchCapability("rv64gc", "lp64d", SourceFmaBatchContract(8, *([True] * 9)))


@pytest.mark.parametrize(
    "change",
    [
        {"isa": "unknown"},
        {"abi": "unknown"},
        {"ieee_f32": False},
        {"gradual_underflow": False},
        {"stable_rounding": False},
        {"nontrapping_unobserved_flags": False},
        {"ieee_f32": 1},
    ],
)
def test_pair_without_explicit_original_capability_refuses(change):
    with pytest.raises(ValueError):
        replace(PAIR, **change).header()


@pytest.mark.parametrize(
    "field",
    [
        "fused_single_rounding",
        "operand_order_preserved",
        "finite_operands_and_results",
        "private_disjoint_storage",
        "stable_rounding",
        "gradual_underflow",
        "nontrapping",
        "exception_flags_unobserved",
        "errno_unobserved",
    ],
)
def test_batch_cannot_drop_an_original_source_or_effect_permission(field):
    with pytest.raises(ValueError, match="contract"):
        replace(BATCH, contract=replace(BATCH.contract, **{field: False})).header()


@pytest.mark.parametrize("name", ["", "1bad", "bad-name", "x;bad", "\u00e9"])
def test_caller_symbol_binding_cannot_inject_source(name):
    for selection in (PAIR, BATCH):
        with pytest.raises(ValueError, match="identifier"):
            selection.header(namespace=name)
    with pytest.raises(ValueError, match="identifier"):
        emit_fixed_outward_f64_header(name=name, host_isa="rv64gc")


@pytest.mark.parametrize("flag", ["narrow_f32", "exact_bound_f32"])
def test_directed_narrowing_cannot_be_selected_by_non_boolean_metadata(flag):
    with pytest.raises(ValueError, match="boolean"):
        emit_fixed_outward_f64_header(name="bound", host_isa="rv64gc", **{flag: 1})


@pytest.fixture
def native_tools():
    compiler = os.environ.get("MERLIN_TEST_RISCV_GCC")
    if not compiler:
        pytest.skip("requires explicitly selected host CPU compiler")
    cc = Path(compiler).resolve(strict=True)
    disassembler = cc.with_name("riscv64-unknown-elf-objdump")
    if not disassembler.is_file():
        pytest.skip("requires selected host toolchain disassembler")
    return cc, disassembler


def _native_source(owner, prefix=""):
    (owner / "pair.h").write_text(PAIR.header())
    (owner / "batch.h").write_text(BATCH.header())
    (owner / "outward.h").write_text(
        emit_fixed_outward_f64_header(
            name="bound",
            host_isa=PAIR.isa,
            narrow_f32=True,
            exact_bound_f32=True,
        )
    )
    source = owner / "probe.c"
    source.write_text(
        prefix
        + """
#include "pair.h"
#include "batch.h"
#include "outward.h"
void pair(float a,float b,float c,float d,float e,float f,float *l,float *h) {
  MERLIN_SOURCE_F32_FMA_PAIR(a,b,c,d,e,f,l,h);
}
void batch(const float *fraction,float *product,float coefficient) {
  MERLIN_SOURCE_F32_FMA_EIGHT(fraction,product,coefficient);
}
double up(double a,double b){return MERLIN_F64_OUTWARD_ADD_UP(a,b);}
double down(double a,double b){return MERLIN_F64_OUTWARD_ADD_DOWN(a,b);}
double product(double a,double b){return MERLIN_F64_OUTWARD_MUL_UP(a,b);}
float cast_up(double a){return MERLIN_F32_EXACT_CEIL_FROM_F64(a);}
float cast_down(double a){return MERLIN_F32_EXACT_FLOOR_FROM_F64(a);}
"""
    )
    return source


def _compile(cc, source):
    return subprocess.run(
        [
            str(cc),
            "-std=c11",
            "-O2",
            "-march=" + PAIR.isa,
            "-mabi=" + PAIR.abi,
            "-c",
            str(source),
            "-o",
            str(source.with_suffix(".o")),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_actual_shared_cpu_object_preserves_fma_lanes_and_fixed_rounding(tmp_path, native_tools):
    cc, objdump = native_tools
    source = _native_source(tmp_path)
    result = _compile(cc, source)
    assert result.returncode == 0, result.stderr
    listing = subprocess.check_output([str(objdump), "-d", str(source.with_suffix(".o"))], text=True, timeout=30)
    instructions = []
    for line in listing.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[0].endswith(":"):
            try:
                int(fields[1], 16)
            except ValueError:
                continue
            instructions.append((fields[2], "".join(fields[3:])))
    assert sum(name == "fmadd.s" for name, _ in instructions) == 2 + BATCH.contract.lanes
    assert [
        (name, operands.split(",")[-1]) for name, operands in instructions if name in ("fadd.d", "fmul.d", "fcvt.s.d")
    ] == [
        ("fadd.d", "rup"),
        ("fadd.d", "rdn"),
        ("fmul.d", "rup"),
        ("fcvt.s.d", "rup"),
        ("fcvt.s.d", "rdn"),
    ]
    assert not any(name.startswith("csr") for name, _ in instructions)


@pytest.mark.parametrize(
    "prefix,diagnostic",
    [
        ("#define MERLIN_ORDERED_FMA_BOUNDS_H\n", "must precede"),
        ("#define MERLIN_F64_OUTWARD_ADD_UP(a,b) ((a)+(b))\n", "already selected"),
    ],
)
def test_actual_host_compilation_refuses_late_or_duplicate_directed_selection(
    tmp_path,
    native_tools,
    prefix,
    diagnostic,
):
    result = _compile(native_tools[0], _native_source(tmp_path, prefix))
    assert result.returncode != 0 and diagnostic in result.stderr
