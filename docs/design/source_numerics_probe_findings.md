---
title: "Source-numerics probe findings: whole-model gates, accumulation order and host compile"
kind: design
status: current
owner: core
last_verified: 2026-10-08
related: [ordered_fma_certificates, optional_passes, source_attention_bound_strategies]
code_refs:
  - src/merlin/llvmlower/cpu_f32_matmul.py
  - src/merlin/llvmlower/ordered_fma_matmul.py
  - src/merlin/llvmlower/cpu_sum.py
  - src/merlin/llvmlower/cpu_flash_exp.py
  - src/merlin/llvmlower/llvm_loop_outline.py
  - src/merlin/llvmlower/normalization_reassociation.py
  - src/merlin/runtime/backends/spike_model.py
---

# Source-numerics probe findings

On 2026-10-05 a set of probes ran a 500M-parameter vision-language-action model (SmolVLA, 1,600
outputs) through Merlin's lowering and checked each candidate against the unchanged elementwise
gate (`atol = 0.03125`, `rtol = 0.02`). Unless a finding says otherwise, the original checkpoint,
weights, inputs and golden output were the same for every run. The receipts were point-in-time
probe directories and are not kept in the tree. This page keeps what they established. None of
it is a target cycle count or a hardware result. Native means the complete lowered LLVM run on
the host with scalar stand-ins for the device.

## Accumulation order decides the gate

* **Source order passes bit-exactly.** The model's attention products come from PyTorch's batched
  `cblas_sgemm_batch` path. An increasing-K fused accumulation from `+0` reproduced all 40 recorded
  f32 attention products, 5,104,560 values in total. A separate multiply and add did not. With that
  order on the 16 actor f32 QK/PV sites, all 1,600 native outputs were bit-identical to the golden
  output. The builder is `cpu_f32_matmul.py` (policy `torch_2_10_cpu_sgemm_ordered_fma`):
  explicit, default off, and it grants no reassociation.
* **Reassociating attention fails the gate.** Changing only the 384 BF16 vision-attention dot
  reductions to f64 fused accumulation followed by f32 narrowing failed **144 of 1,600** outputs
  (max abs error 0.143, relative L2 0.024). Changing only the 96 QK reductions failed 168, and
  changing only the 288 PV reductions failed 123. Accumulation order is a material accuracy risk in
  both stages. An exact integer reconstruction of a dot product cannot be presumed to preserve a
  floating source reduction.
* **Normalization sums need the source cascade.** Before a fresh SiLU capture and the source-order
  RMS sum, the gate failed 69 outputs. The first divergence was layer 1's RMSNorm: its input was exact but six BF16 outputs
  differed, and a serial f32 sum reproduced exactly those six. PyTorch's AVX2 sum cascade matched all
  226 recorded means of layers 0 and 1. Multiplying by a reciprocal instead of dividing by the width
  changed 79 and 78 of those means and was rejected. With the cascade (`cpu_sum.py`, policy
  `torch_2_10_cpu_sum_f32_avx2`, widths 8 to 8191), 3 of 1,600 outputs still failed (max abs error
  0.050, relative L2 0.0090), until the source-order attention above closed the gap.

The pinned PyTorch exp kernel (`cpu_flash_exp.py`, `fexp_u20`) was checked exhaustively over all
1,118,743,633 binary32 inputs from its underflow threshold to negative zero, with zero monotonic
decreases. That holds only in the recorded IEEE round-to-nearest-even, gradual-underflow
environment with the exact source polynomial and FMA sequence. It is not a licence for a generic
exp approximation.

Consequence for optional passes: a numerics-changing pass that reorders reductions, such as
`layer-norm-chunked-sums`, has to be graded against the unchanged whole-model gate, never only
against its own unit reference.

## Host compile time is a loop-extraction problem

Plain O2 compilation of the fused or unfused host module timed out at 900 seconds. LLVM loop
extraction (`llvm_loop_outline.py`, the default-off `outline_llvm_loops` feature) produced 7,617
helpers, kept the 3 original functions, and shrank the main function from 506,699 to 147,641
LLVM lines. The corrected policy compiled at O2 in **111.254 seconds**, and all 1,600 native
outputs stayed bit-exact. Helper call overhead was not measured.

Two attribute traps showed up on the way. `forceattrs` does not add `noinline` to a function that still carries
`alwaysinline`, and removing the conflicting attribute in the same pass is too late. The policy
must remove `alwaysinline` in one completed pass and then add `noinline`. Also, `llvm-diff`
ignores function attributes, so it cannot show that two emitted policies are the same. The
regression test checks every extracted helper's attributes directly.

## Mixed compilers corrupt the BF16 calling convention

On the pinned RV64GC LP64D tools, Clang passes and returns BF16 in floating-point registers.
GCC's `unsigned short` fallback uses integer registers. The model was compiled with Clang and the
MLIR runtime helpers with GCC, so every BF16 helper call crossed the convention. On Spike, an
optimized prototype failed 1,560 of 1,600 outputs, with all output rows repeating. The fix
compiles the shared MLIR helper unit with the model's Clang against the headers of GCC's
reported sysroot (`spike_model.py`) and records the command in the build identity.

## Exact replay and bound screens on the first vision attention block

These ran on CPU emulations of the device integer planes, preserving the original ordered source
replay. Every control returned all 786,432 original BF16 output bits with no false certificate.
The percentages are the fraction of outputs replayed in source order. The counts are arithmetic
and traffic, not cycles.

| Bound | QK replay | PV replay | Scalar replay MAC | Int8 MAC | Readback bytes |
|---|---:|---:|---:|---:|---:|
| Selective denominator contributions | 16.66% | 26.51% | 347,600,896 | | |
| Monotone source exp endpoints | 12.98% | 26.03% | 314,172,864 | | |
| Absolute digit norm bound | 4.51% | 8.92% | 108,148,544 | 28,991,029,248 | 1,019,215,872 |
| Signed-prefix QK, original PV | 2.42% | 8.71% | 89,619,392 | 28,991,029,248 | 1,019,215,872 |
| Signed-prefix QK, source PV 192/192/128 | 2.42% | 4.23% | 53,532,608 | 28,991,029,248 | 1,245,708,288 |
| Signed-prefix QK, PV in 64-chunks | 2.42% | 2.35% | 38,396,864 | 28,991,029,248 | 1,811,939,328 |
| Holder metadata + source PV parts | 3.45% | 10.48% | 112,192,640 | 14,495,514,624 | 622,854,144 |

* Tighter bounds cut scalar replay, but each paid for it in integer work or readback. The
  absolute-norm bound doubles integer work. PV in 64-chunks adds 792.7 MB of readback and was not
  promoted. Holder metadata halves the integer MACs and cuts readback by 38.89%, but replays more.
* Generic exact FMA replay capsules (`ordered_fma_matmul.py`: independent outputs, each scalar
  chain kept intact) ran at
  10.19 to 10.50 Spike instructions per MAC with one output per group, and at 5.65 to 5.88 with
  four. Even at the better rate, the model's replay count implies more than 7 billion scalar
  instructions before any other work. Replay volume, not bound arithmetic, was the limit.

Bound selection stays default off. Promoting one needs numeric, order and range legality plus a
measured end-to-end target cost. The variants that were built and dropped are recorded in
`source_attention_bound_strategies.md`.
