# Rocket with FP32 Gemmini

The descriptor in `target/descriptor.yaml` selects a distinct FP32 target. Its
contract describes `chipyard.FPGemminiRocketConfig`; generated facts supply the
actual operand/weight/accumulator types, geometry and instruction encodings.
The reviewed integer target and hardware pin registry remain separate.

For a Core ATen recipe, select:

```yaml
core_aten:
  placement:
    policy: supported_contractions_v1
    rtl_facts: /selected/fp32/facts.json
```

Keep the existing overlay and public/hidden capture selections; omit authored
row `lane_expectation` when using derived placement. The stage assigns `device`
when captured IR contains a contraction supported by those datapath facts and
`host` otherwise. Facts bytes enter the derivation receipt and exported corpus;
changed facts require rederivation. Device passes require retained executed
instruction evidence; host passes require the host lane. Both are scored.

The target-owned provider uses the submitted `mlir_oot.golden_device_catalog`
API and the isolated `out/build/fp32-gemmini/lib/libgemmini.so` extension.
Select `MERLIN_OUT_ROOT` for the operator output root, `MERLIN_RTL_FACTS` for the
exact extraction, and optionally `MERLIN_FP32_SPIKE_EXTLIB` for another isolated
FP32 build. Missing inputs refuse execution. The existing staging helper
`examples/gemmini/core_aten/stage_inputs.py` accepts `--definition
examples/gemmini_fp32` and explicit facts/headers/corpus paths.

`experiment.yaml` is an unprepared definition: its release paths must be
replaced through genuine Phase 0 and reviewed corpus preparation. These files
neither seal a corpus nor launch Phase 1. L2 evidence is functional simulation;
RTL certification and a reviewed FP32 hardware pin are separate prerequisites.
