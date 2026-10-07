"""Closed batches retain their readout and call an independently generated matrix kernel.

These are synthetic dimensions, not validation-model forms. No target kernel is
implemented here: the route must state the per-slice program and its separate ABI.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest
from fake_quant_layer import Oracle

from merlin.common import mlir_query as mq
from merlin.llvmlower.device_offload import rewrite_groups_to_device
from merlin.xdsl_dialects.lowering import compute_groups as CG
from merlin.xdsl_dialects.lowering import group_command as GC


def _source(batches=(3,), *, rhs_batch_axis=0):
    args, outputs, lines = [], [], []
    for i, batch in enumerate(batches):
        lhs, rhs, out = f"{batch}x2x5", f"{batch}x5x4", f"{batch}x2x4"
        args += [f"%a{i}: tensor<{lhs}xi8>", f"%b{i}: tensor<{rhs}xi8>"]
        outputs.append(f"tensor<{out}xi8>")
        lines += [
            f'%ad{i} = "quant_ext.dequantize_per_tensor"(%a{i}, %s, %z) '
            f"<{{quant_min = -128 : i64, quant_max = 127 : i64}}> : "
            f"(tensor<{lhs}xi8>, tensor<f32>, tensor<i64>) -> tensor<{lhs}xf32>",
            f'%bd{i} = "quant_ext.dequantize_per_tensor"(%b{i}, %s, %z) '
            f"<{{quant_min = -128 : i64, quant_max = 127 : i64}}> : "
            f"(tensor<{rhs}xi8>, tensor<f32>, tensor<i64>) -> tensor<{rhs}xf32>",
            f"%e{i} = tensor.empty() : tensor<{out}xf32>",
            f"%f{i} = linalg.fill ins(%zero : f32) outs(%e{i} : tensor<{out}xf32>) -> tensor<{out}xf32>",
            f"%mm{i} = linalg.generic {{indexing_maps = [",
            "affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,",
            f"affine_map<(d0, d1, d2, d3) -> (d{rhs_batch_axis}, d3, d2)>,",
            "affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],",
            'iterator_types = ["parallel", "parallel", "parallel", "reduction"]}',
            f"ins(%ad{i}, %bd{i} : tensor<{lhs}xf32>, tensor<{rhs}xf32>) outs(%f{i} : tensor<{out}xf32>) {{",
            "^bb0(%x: f32, %y: f32, %acc: f32):",
            "%product = arith.mulf %x, %y : f32",
            "%sum = arith.addf %product, %acc : f32",
            "linalg.yield %sum : f32",
            f"}} -> tensor<{out}xf32>",
            f'%q{i} = "quant_ext.quantize_per_tensor"(%mm{i}, %s, %z) '
            f'<{{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}}> : '
            f"(tensor<{out}xf32>, tensor<f32>, tensor<i64>) -> tensor<{out}xi8>",
        ]
    return "\n".join(
        [
            "builtin.module {",
            f"func.func @forward({', '.join(args)}) -> ({', '.join(outputs)}) {{",
            "%s = arith.constant dense<0.5> : tensor<f32>",
            "%z = arith.constant dense<0> : tensor<i64>",
            "%zero = arith.constant 0.0 : f32",
            *lines,
            f"func.return {', '.join(f'%q{i}' for i in range(len(batches)))} : {', '.join(outputs)}",
            "}",
            "}",
        ]
    )


def test_closed_batch_is_one_matrix_program_per_slice_with_the_original_readout():
    module = mq.parse(_source())
    (group,) = [g for g in CG.form_groups(module, "synthetic", oracle=Oracle()) if g.placement != CG.HOST]
    stated = GC.program(group, weight_args=())
    assert (stated.entry["M"], stated.entry["K"], stated.entry["N"]) == (2, 5, 4)
    assert stated.entry["epilogue"] == ["acc_scale"]
    assert stated.entry["acc_scale"] == 0.5
    assert stated.batch_shape == (3,)
    assert stated.to_dict()["batch_shape"] == [3]


def test_distinct_batch_counts_have_distinct_typed_calls_to_the_same_slice_program(tmp_path):
    module = mq.parse(_source((3, 7)))
    result = rewrite_groups_to_device(
        module, "synthetic", oracle=Oracle(), weight_args=(), select=lambda _s: True, sidecar_dir=tmp_path
    )
    assert result.moved == 2, result.skipped
    assert set(result.signatures.values()) == {(3, 2, 4, 5), (7, 2, 4, 5)}
    assert len(result.entries) == 2
    assert len({row.symbol for row in result.routed}) == 2
    assert mq.op_count(module, "func.call") == 2
    assert mq.op_count(module, "linalg.generic") == 0
    assert all(entry["M"] == 2 and entry["acc_scale"] == 0.5 for entry in result.entries.values())
    module.verify()


def test_a_different_rhs_batch_coordinate_is_not_restated_as_disjoint_slices():
    module = mq.parse(_source((3,), rhs_batch_axis=1))
    groups = [g for g in CG.form_groups(module, "synthetic", oracle=Oracle()) if g.placement != CG.HOST]
    assert groups
    with pytest.raises(CG.NoCapsuleForm, match="batch|coordinate|orientation"):
        GC.program(groups[0], weight_args=())


@pytest.mark.parametrize("stored", [(), (1,)])
def test_immutable_and_runtime_rhs_use_the_same_batch_slice_program(stored):
    module = mq.parse(_source())
    (group,) = [g for g in CG.form_groups(module, "synthetic", oracle=Oracle()) if g.placement != CG.HOST]
    stated = GC.program(group, weight_args=stored)
    assert stated.stored_operand is None  # no prepack or guessed orientation
    assert stated.batch_shape == (3,)


def test_a_batch_command_does_not_discard_nonzero_accumulator_initialization():
    module = mq.parse(_source().replace("%zero = arith.constant 0.0", "%zero = arith.constant 1.0"))
    (group,) = [g for g in CG.form_groups(module, "synthetic", oracle=Oracle()) if g.placement != CG.HOST]
    with pytest.raises(CG.NoCapsuleForm, match="accumulator seed"):
        GC.program(group, weight_args=())


def test_a_command_buffer_cannot_silently_execute_only_one_batch_slice():
    from merlin.llvmlower import whole_program as WP

    with pytest.raises(WP.WholeProgramError, match="batch.*slice"):
        WP.whole_program_buffer(
            mq.parse(_source()),
            "synthetic",
            weight_args=(),
            oracle=Oracle(),
            manifest={"0": {"kind": "input", "name": "left"}, "1": {"kind": "input", "name": "right"}},
        )


def test_closed_batch_route_builds_actual_generated_device_objects(tmp_path):
    from pathlib import Path

    from merlin.llvmlower.device_build import FROM_GROUP, build_device_objects
    from merlin.llvmlower.toolchain import llvm_nm
    from merlin.mining.registry import load_rvv_package
    from merlin.runtime.backends.spike_model import selected_model_compiler_plan

    package = os.environ.get("MERLIN_TEST_DEVICE_PACKAGE")
    if not package:
        pytest.skip("select the generated device package to qualify actual batch objects")
    host = os.environ.get("MERLIN_TEST_HOST_PACKAGE")
    assert host, "actual batch qualification needs the selected host package, not a default ISA"
    selected_host = load_rvv_package(Path(host))
    host_plan = selected_model_compiler_plan(
        backend=selected_host.backend,
        cflags_override=list(selected_host.cflags),
        vlen=None,
        features=frozenset(),
    )
    assigned = rewrite_groups_to_device(
        mq.parse(_source((3, 7))), "gemmini", oracle=Oracle(), weight_args=(), select=lambda _s: True
    )
    assert assigned.moved == 2
    assert all(name == "prepack" for name, _reason in assigned.skipped)
    built = build_device_objects(
        "gemmini",
        assigned.signatures,
        {row.symbol: row.dtypes for row in assigned.routed},
        package_dir=package,
        workdir=tmp_path / "device",
        operand_dtype="int8",
        accum_dtype="i32",
        entries=assigned.entries,
        cflags=host_plan["observation"]["cross_flags"],
    )
    assert built.ok and not built.skipped, built.skipped
    assert set(built.kernels) == set(assigned.signatures)
    assert set(built.built_from.values()) == {FROM_GROUP}
    archive = built.archive(tmp_path / "device.a")
    assert archive is not None and archive.stat().st_size > 0
    nm = subprocess.run([str(llvm_nm()), "--defined-only", str(archive)], capture_output=True, text=True, timeout=10)
    assert nm.returncode == 0, nm.stderr
    symbols = {line.split()[-1] for line in nm.stdout.splitlines() if line.strip() and not line.endswith(":")}
    assert set(built.kernels) | set(built.kernels.values()) <= symbols
    interfaces = sorted(Path(tmp_path / "device").glob("*.iface.mlir"))
    # Each public ABI keeps its own interface artifact, even if a builder later
    # shares an identical per-slice kernel object across the two callers.
    assert len(interfaces) == 2


_DRIVER = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
enum { B=3, M=$M, N=4, K=$K, MP=4, NP=4, KP=$KP };
static int calls;
/* Independent portable stand-in for the GENERATED device artifact, only in this test. */
void test_kernel(void *weight, void *lhs, void *out) {
  puts("called"); ++calls;
  const int8_t *a=lhs, *b=weight; int32_t *c=out;
  for (int i=0;i<MP;++i) for (int j=0;j<NP;++j) {
    int32_t sum=0;
    for (int p=0;p<KP;++p) sum+=(int32_t)a[i*KP+p]*b[p*NP+j];
    c[i*NP+j]=sum;
  }
}
int main(int argc, char **argv) {
  struct rlimit no_core={0,0};
  if (setrlimit(RLIMIT_CORE,&no_core)) return 7;
  setbuf(stdout,0);
  int mode=argc>1 ? atoi(argv[1]) : 0;
  static int8_t A[B*M*K+3], W[B*K*N+5], old_A[sizeof A], old_W[sizeof W];
  static int32_t C[B*M*N+7];
  for (int i=0;i<B*M*K;++i) A[i+3]=(i%5)-2;
  for (int i=0;i<B*K*N;++i) W[i+5]=(i%7)-3;
  for (unsigned i=0;i<sizeof C/sizeof *C;++i) C[i]=123456;
  memcpy(old_A,A,sizeof A); memcpy(old_W,W,sizeof W);
  void *a_ptr=A, *b_ptr=W, *c_ptr=C;
  intptr_t a_off=3, c_off=7, c_stride=M*N;
  if (mode==1) { c_ptr=A; c_off=0; }
  if (mode==2) c_stride=M*N-1;
  if (mode==3) a_off=-1;
  if (mode==4) b_ptr=0;
  if (mode==5) a_off=INTPTR_MAX;
  merlin_memref_3d r=entry(A,a_ptr,a_off,B,M,K,M*K,K,1,
                         W,b_ptr,5,mode==6 ? B-1 : B,K,N,K*N,N,1,
                         C,c_ptr,c_off,B,M,N,c_stride,N,1);
  if (mode==6) return calls==0 && !r.aligned ? 0 : 1;
  if (mode) return 2; /* all other invalid descriptors must trap before the kernel */
  if (calls!=B || r.aligned!=C || r.offset!=7) return 3;
  for (int b=0;b<B;++b) for (int i=0;i<M;++i) for (int j=0;j<N;++j) {
    int32_t sum=0;
    for (int p=0;p<K;++p) sum+=(int32_t)A[3+b*M*K+i*K+p]*W[5+b*K*N+p*N+j];
    if (C[7+b*M*N+i*N+j]!=sum) return 4;
  }
  for (int i=0;i<7;++i) if (C[i]!=123456) return 5;
  return memcmp(A,old_A,sizeof A) || memcmp(W,old_W,sizeof W);
}
"""


@pytest.mark.parametrize(("rows", "reduced"), [(4, 4), (2, 5)])
def test_compiled_batch_adapter_checks_all_values_offsets_and_invalid_descriptors(tmp_path, monkeypatch, rows, reduced):
    from merlin.llvmlower import device_shim as DS

    cc = shutil.which("cc")
    assert cc is not None, "this ABI qualification needs an actual C compiler"
    monkeypatch.setattr(DS, "kernel_abi_for", lambda _device: DS.KernelAbi("test_kernel"))
    unit = DS.emit_translation_unit(
        "synthetic", {"entry": (3, rows, 4, reduced)}, {"entry": ("i8", "i8", "i32")}, tile_edge=4
    )
    assert unit.symbols == ("entry",), unit.skipped
    driver = _DRIVER.replace("$M", str(rows)).replace("$KP", str((reduced + 3) // 4 * 4)).replace("$K", str(reduced))
    source, executable = tmp_path / "adapter.c", tmp_path / "adapter"
    source.write_text(unit.text + driver)
    built = subprocess.run(
        [cc, "-std=c11", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(executable)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert built.returncode == 0, built.stderr
    for mode in range(7):
        result = subprocess.run([str(executable), str(mode)], capture_output=True, text=True, timeout=10)
        if mode in (0, 6):
            assert result.returncode == 0, (mode, result.stdout, result.stderr)
        else:
            assert result.returncode < 0, (mode, result.returncode)
            assert "called" not in result.stdout, "invalid descriptors must not reach the device"
