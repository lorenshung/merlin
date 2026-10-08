# Track A Journal — BF16→FP8 quantization (Pi0 / SmolVLA on Atlas)

Working log for MERLIN_PLAN.md Track A. Each entry records **what changed**, **what was tested**, and
**the verified outcome**. "Verified" means a command was run and its output observed; anything not run is
marked *not verified*. Commits live on branch `track-a` (listed at the end).

Baseline: merlin `183cbb09f2cd` (main), host `garden` for entries E1–E21. On 2026-10-07 the work was moved onto
branch `track-a`, rebased onto upstream `4a7cb1821` and re-verified there (E22).

---

## 2026-10-01

### E1. Environment setup
- **Changed:** created `.venv/` (untracked) via the documented base install:
  `uv venv && uv pip install -e '.[dev,xdsl,targetgen]'` and
  `uv pip install -e packages/merlin-experiments -e packages/merlin-dse -e packages/merlin-mining -e packages/merlin-analysis`.
- **Tested:** `.venv/bin/python -c "import merlin"`.
- **Verified outcome:** imports from `src/merlin/__init__.py`. ✅
- **Observed gaps (no change):** no model2MLIR checkout, no `MERLIN_*` env vars, torch/torchao not
  pinned in `pyproject.toml`, HF cache `/scratch2/loren/hf_cache` has no Pi0/SmolVLA weights.
  GPU is TITAN RTX (SM 7.5) → no native FP8 `_scaled_mm`; FP8 numerics need a CPU/emulated reference.

### E2. VCS availability
- **Changed:** nothing in repo. Scratch test only.
- **Tested:** `source /ecad/tools/vlsi.bashrc; vcs -full64 -sverilog t.sv -o simv; ./simv`
  on a one-line `$display` module.
- **Verified outcome:** VCS V-2023.12-SP1-1 licensed; compiled and printed `VCS_OK`. ✅

### E3. Atlas hardware sources located
- **Changed:** nothing.
- **Found:** local atlas-npu checkout at `/scratch2/loren/bringup-chipyard/generators/atlas-npu`,
  HEAD `0079c05` = merlin pin `atlas_npu` `569b7c31` + a LICENSE-only commit
  (`git diff --stat 569b7c3 0079c05` → `LICENSE | 29 +`). `dependencies/sp26-fp-units` at `9a0cc09`
  matches pin `atlas_fp_units`. Submodules include `pi0-quant` (`8bc0f6a`) and `npu-model` (`11598ec`).
- **Verified outcome:** revisions match the pins by commit; content digests of the locally-edited
  sources *not yet verified* (in progress).

### E4. Atlas numerical contract audit (read-only)
- **Finding:** `examples/atlas/target/software-spec.yaml` fully declares the *hardware* numerics
  (OCP E4M3FN operands, BF16 per-step RNE accumulate, subnormal flush, exponent range [1,14] → qualified
  |x| ≤ 240, finite BF16 clamp) but leaves the *model* quantization policy unknown:
  `quantization.formats[0].scale_encoding: unknown`, `block_size: unknown` (lines 53-54). vmatmul applies
  no scale; E8M0 registers belong to pack/pop, not to model operands.
- **Conflicts spotted (need architect ruling):** vpack multiplies by scale but vmatpop divides; the ISA
  reference model accumulates in fp16 and rounds once vs the spec's per-step BF16; vmatpop uses
  `torch.div(..., rounding_mode="trunc")`.

### E5. A1 — one contraction traced source → target (read-only)
- **Changed:** nothing tracked. Scratch venv (xdsl 0.68.0, triton 3.7.1) for the walkthrough.
- **Tested:** `examples/triton/walkthrough.py --package out/artifacts/targets/gemmini/hand_v0` → all 9 steps
  pass; command buffer `RES_PACK W, MATMUL_RESIDENT×2, COMMIT×2, EVICT`; outputs match numpy (L0, software
  simulator only). Hand-written probe modules through `compile_core`.
- **Verified outcome:** ✅ A1 gate met for the example stack (gemmini `hand_v0`; ToyNPU is docs-only).
  Observed by probe: a `linalg.transpose`-fed `A@Wᵀ` is refused at `interface_lowering.py:827`; a dequantized
  pack is refused at `target_lowering.py:242`; xDSL cannot build `linalg.matmul` over `f8E4M3FN`.
- **Atlas:** no staged route (compiler is the agent-generated OOT backend); `vmatmul` takes no scale; E8M0
  registers only feed pack/pop. ⇒ model-operand scales must be applied post-accumulation in BF16 (VPU/host),
  valid for per-tensor / per-output-channel scales, **not** for scales varying along K.
- **Metadata-loss boundaries found (9):** TorchAO config (B1/B2), weight-only capture decodes FP8 to f32 args,
  TorchAO subclass scales only in `qinner::` sidecar, no dtype gate at routing + capacity counts floats as 32-bit,
  no FP8 linalg in xDSL, transpose permutation not recorded / scale axis not remapped, no scale slot on
  matmul/commit, no fp8 runtime dtype token or `scale` role emitted, Atlas scale encoding unknown.

### E6. TorchAO behaviour on this host (scratch venv, torch 2.10.0+cpu, torchao 0.17.0 = model2MLIR lock)
- **Tested:** `quantize_` with `Float8DynamicActivationFloat8WeightConfig` on bf16 `Linear(64,32)`.
- **Verified outcome:** `quantize_` refuses on SM 7.5 (`requires CUDA compute capability ≥8.9`); with the
  hardware check monkeypatched (structure-only), weight becomes `Float8Tensor`; CPU forward works for
  **PerTensor** and fails for **PerRow** on torch 2.10 (`_scaled_mm only supports per-tensor scaling for CPU`).
- **model2MLIR:** only checkout on host is `/scratch2/mahsa/model2MLIR` (not ours); it lacks the bundle/receipt
  and `quantization_preapplied` APIs the current worker requires → capture of the mixed MLP fails. A newer
  model2MLIR revision is needed and is not pinned anywhere in merlin.

### E7. Fix B1+B2 — dynamic TorchAO config honours the recipe's dtypes and per-tensor granularity
- **Changed:** `src/merlin/targetgen/_recipe_quantizer.py` `quantize_config`: passes `activation_dtype`,
  `weight_dtype` explicitly; reads activation granularity from the activation spec; passes
  `granularity=(act, weight)`; refuses mixed formats, unknown granularities, mixed float8 granularities and
  per-tensor int8 activations with a named `RecipeError`.
- **Added:** `merlin/tests/targetgen/test_recipe_quantizer_dynamic.py` (9 tests; stubs torch/torchao and
  records constructor kwargs).
- **Tested:**
  1. new tests with fix → `9 passed`; same tests against baseline source → `8 failed, 1 passed` (they detect the bug).
  2. real torchao 0.17.0 (`scratchpad/verify_b1b2.py`): baseline turns an `fp8_e5m2` recipe into
     `weight.qdata=float8_e4m3fn, act=float8_e4m3fn`; fixed gives `float8_e5m2` for both. E4M3 per-tensor
     CPU forward → bf16 `(4,32)` OK.
  3. quant regression set (12 files) → `88 passed, 7 skipped, 1 failed`; the failure
     (`test_declared_numeric_profile[mx_gemmini]`, "requires an explicitly selected OOT support provider")
     is pre-existing and unrelated. `ruff check` clean.
- **Verified outcome:** ✅ recipe dtype/granularity now reach TorchAO exactly.

### E8. Fix B3+B4 — capture worker refuses unknown dtypes; dynamic FP8 scheme recorded
- **Changed:** `src/merlin/targetgen/_m2m_capture_worker.py`: argument check refuses a `--dtype` absent from
  `_SCHEME` when neither `--recipe`, `--scheme` nor `--already-quantized` decides; `realized_scheme` gains the
  `quantize_` branch for `fp8_*_dynamic_act_weight`. The `fp8 → float8_weight_only_e4m3` default is left
  unchanged on purpose (existing whole-model recaptures depend on it; the meta already records that scheme name).
- **Added:** `merlin/tests/targetgen/test_capture_dtype_is_known.py` (3 tests).
- **Tested:** new tests → `3 passed`; baseline worker lets `--dtype fp8_e5m2` past the arg check to the m2m
  import (i.e. it would capture fp32). Capture-worker regression set → `94 passed, 19 skipped, 1 failed`
  (`test_capsule_source::test_args_from_cb_linalg_positional`: `KeyError: 'muon'` — OOT backend absent,
  unrelated).
- **Verified outcome:** ✅ B3 (unknown dtype) verified. B4 is *not verified end-to-end* (needs a working
  model2MLIR capture); key names checked against `_recipe_quantizer.py:691,695`.

### E9. A5 (hardware semantics) — Atlas MXU RTL under VCS is bit-exact vs the pi0-quant C model
- **Changed:** nothing in merlin source or the shared atlas-npu checkout. Work done in an rsync copy
  (`scratchpad/atlas-npu-copy`, source bytes hash-identical to the original).
- **Hardware revision:** atlas-npu `0079c05` (= pin `569b7c31` + LICENSE), sp26-fp-units `9a0cc09`, fpex `ab0cf6f`,
  pi0-quant `8bc0f6a`, npu-model `11598ec`. **Clean tree** — the pin's `local_edits` digests do not match;
  `provenance.verify`: `atlas_fp_units` ok, `atlas_npu` drift (commit/branch/origin spelling). Recorded in the
  artifact's `provenance.json` `gaps`.
- **Tested (VCS V-2023.12-SP1-1 via chisel svsim, mill 0.13.0-M0):**
  - `./mill -i atlas.test.testOnly atlas.sa.SystolicArrayTest` → PASS (10 sub-checks, 22 s).
  - `atlas.ipt.InnerProductTreesTest` → PASS (5 configs, 32 s).
  - `SystolicArrayFullMatmulTest` / `SystolicArrayVectorE4M3Test` with new vectors generated from the unmodified
    pi0-quant C headers (ctypes): 12 full-matmul cases (random, exponent-15, ±448, signed zeros, subnormal bytes,
    0x7F/0xFF, cancellation, tiny exponents, K=128/256 chained) + 12 E4M3-output cases (scale_exp −14..+6).
  - Compared per-element RTL dumps bit-for-bit (`compare.py`; the tests' 1 % tolerance not used).
- **Verified outcome:**
  | Comparison | Elements | Bit mismatches |
  |---|---|---|
  | mxu0 SA full matmul (BF16 out) vs SA C model | 12,288 | **0** ✅ |
  | mxu1 IPT full matmul (BF16 out) vs IPT C model | 12,288 | **0** ✅ |
  | Negative control: IPT RTL vs SA C model | 12,288 | 7,069 (checker discriminates) ✅ |
  | SA E4M3 output path (vmatpop.fp8, scaled) | 12,288 | 32 — all lane 5 of case `bias_sm2_specials` |
  - The 32: bias byte 0xFF via PushAccFP8 — RTL `FPUtils.e4m3ToBf16` flushes it to signed zero, C model
    decodes −480. As a PE *operand* 0x7F/0xFF is finite ±480 in both (case `nan_encoding_7f` bit-exact): the RTL
    is internally inconsistent on this encoding. Only reachable with a NaN-encoded bias byte.
  - **Exponent-15 operands are finite normals in RTL** (`e15_x_e15` bit-exact) → the spec's `exponent_range:
    [1,14]` is stricter than the hardware (B14). Subnormal operands → zero in both. 
  - The scaled E4M3 pop cases matching confirms RTL multiplies by `2^(code-127)` on vmatpop (evidence for B11).
  - SA and IPT give different bits on the same inputs (7,069/12,288) → the spec must say which MXU (B13).
- **Caveat:** `check_provenance.py` reports `OK (0 verdict-claiming report(s) checked)` — it did not classify
  this artifact as a verdict, so that gate is vacuous here; provenance is recorded by hand in `provenance.json`.
- **Artifact:** `out/artifacts/probes/atlas_mxu_vcs/20261002T054429Z/` (commands.txt, generator/, vectors/,
  rtl_outputs/, logs/vcs/, compare_summary.json, findings.json, provenance.json).

### E10. A5 (accuracy + oracle question) — BF16→FP8 numerics study on Pi0/SmolVLA-shaped layers
- **Changed:** nothing tracked. Scripts + results in the artifact below (pi0-quant C models built in scratch, called
  via ctypes; FP8Pack cast imported read-only from the checkout).
- **Data:** **SYNTHETIC** (no Pi0/SmolVLA weights on host): W~N(0,0.02)+outliers, X~Student-t(3) with outlier
  channels, M=64. Functional models only (RTL equivalence of those models is E9).
- **Cross-checks (all verified bit-exact):** IPT C ≡ IPT python (incl. K tails); SA C ≡ sequential per-element BF16
  RNE accumulation (all matmuls + 4.1 M single steps, 0 mismatches); zero-padding K 4304→4320 and 720→736 changes
  no bits (SA, IPT, torch); real torchao 0.17 CPU per-tensor weight bytes ≡ emulation, outputs 99.97 % bit-equal
  (rest 1 BF16 ulp: torchao adds bias inside the matmul).
- **Verified outcome (rel-L2; (2) = TorchAO-style FP8, fp32 accumulate, pi0-quant po2 scales):**
  | layer (K→N) | BF16 vs (2) | (2) vs mxu0 SA | (2) vs mxu1 IPT | BF16 vs SA | BF16 vs IPT |
  |---|---|---|---|---|---|
  | Gemma-2B q 2048→2048 | 3.8 % | 4.5 % | 1.0 % | 5.9 % | 3.9 % |
  | Gemma-2B down 16384→2048 | 4.0 % | 11.1 % | 2.4 % | 11.8 % | 4.7 % |
  | Expert down 4096→1024 | 3.7 % | 7.2 % | 1.5 % | 8.1 % | 4.0 % |
  | SigLIP fc2 4304→1152 (tail) | 3.8 % | 6.0 % | 1.3 % | 7.1 % | 4.0 % |
  | SmolVLA expert 720→2048 (tail) | 3.9 % | 2.1 % | 0.5 % | 4.4 % | 3.9 % |
  - K sweep: SA error ~√K (0.5 % @32 → 3.2 % @2048 → 10.4 % @16384); IPT 0.02 % → 0.8 % → 2.8 %.
  - ⇒ GPU/TorchAO FP8 is **not** an oracle for mxu0 at K ≥ 2048 (per-step BF16 rounding ≥ quantization error);
    acceptable (~1 %) for mxu1 at K ≤ 2048. Cast differences (torch vs FP8Pack) contribute only 0.06–0.2 %.
  - Scale rule: TorchAO's own fp32 `amax/448` scale differs from po2 by 5–6 % → an oracle must use the hardware's
    po2 scales. `ceil(log2(amax/448))` never clips and is ≤ error of pi0-quant's `floor(log2(amax/256))` (which
    clips in 18.8 % of tensors and always uses exponent 15). Staying in `[1,14]` needs `amax/2^s < 248`.
  - Edge cases: all-zero tensor → naive recipe crashes (`log2(0)`), real torchao → 100 % NaN, pi0-quant guards
    (scale_exp 0). One 30000 outlier → Atlas flushes 66.6 % of the activation (no FP8 subnormals) vs 5 % in torch;
    non-outlier rows 43 % error. NaN/Inf → Atlas silently zeros/saturates (one corrupted row); torchao → all NaN.
    Transposed weight → identical scale, bytes and SA output.
- **Provenance:** `provenance.record()` embedded (`atlas_npu` drift, `atlas_fp_units` ok); `check_provenance.py` →
  `OK (1 verdict-claiming report(s) checked)`.
- **Artifact:** `out/artifacts/probes/atlas_fp8_numerics/20261002T055039Z/` (results.json, summary.md, scripts/).

### E11. A2 — Atlas FP8 policy gaps, written up as questions for the spec owners
- **Ownership note (user, 2026-10-02):** the software spec is owned by the hardware team, not by this work. The
  file below is evidence + open questions to hand them, **not** a proposed spec authored here.
- **Changed:** new untracked `out/artifacts/handoff/track_a/atlas_fp8_policy_proposal.yaml` — every field tagged
  extracted / observed (VCS) / authored (pi0-quant source) / RULING.
- **Tested:** merged into a copy of `examples/atlas/target/software-spec.yaml` and run through
  `validate_software_spec(doc, "atlas")`.
- **Verified outcome:** accepted after dropping `block_size: none` (schema allows only int/unknown). **But** the
  acceptance is too permissive: undeclared keys (`scale_rule`, `scale_application`, `cast_*`) are accepted without
  being checked, and the placeholder `RULING` counts as a known value → the A2 gate ("missing or conflicting
  fields cause explicit refusal") is **not met** by the current schema (B20). The tracked Atlas spec was not edited.

### E12. New inputs from user: atlas-mlir + model2MLIR (A4 unblocked)
- **Changed:** cloned `git@github.com:ucb-bar/atlas-mlir.git` → `/scratch2/loren/atlas-mlir` (HEAD `842bd46`;
  merlin's `target_support.json` references `5485aa0`, one docs commit earlier). Selected RTL in its README is
  atlas-npu `0079c054` — **the same clean revision the VCS run (E9) used**.
  Cloned `git@github.com:ucb-bar/model2MLIR.git` (HEAD `90f53bc`); it has the bundle/receipt and
  `quantization_preapplied` APIs the current capture worker requires (absent from the copy tried in E6).
- **Disk policy:** stay on `/scratch2` (user request; ~45 GB free) — lean LLVM/MLIR build, CPU torch.
- **Environment note:** `cmake` on PATH is a broken Xilinx 3.3.2 (missing `libidn.so.11`); a pip/uv cmake is used.
- **In progress:** (1) m2m capture of coverage_mlp at bf16 / dynamic FP8 / static FP8 + metadata boundary
  trace; (2) LLVM/MLIR `a47bddcc` + atlas-mlir build and test suite; (3) machine-level FP8 metadata trace.

### E13. A4 machine level — what one scaled FP8 linear becomes on Atlas (read-only analysis of atlas-mlir)
- **Found:** atlas-mlir handoff bundles are fixed 32×32 tiles with **unit scales only** (`seli e3,127`; every
  `mxu_pop bf16` has `scale_reg=0`, enforced by the verifier `lib/AtlasOps.cpp:137-138`). Code 127 = unit confirmed
  by `docs/vpu-e8m0-pack-observation.md:9-12` (npu-model's `seli eX,1` is unit only under its wrong model — B12).
- **Rescale placement:** post-pop VPU multiply by a raw-bits `vli all` constant `(127+sx+sw)<<7` — the same
  pattern the attention bundle already executes. Folding into `vmatpop.fp8` is representable but never executed.
- **Dynamic per-tensor activation scaling on-device is expressible** (row/col max reductions → `vstore`/`lhu`/
  `srli 7`/`andi 0xff` → `seld`), but `seld` has never run on the core → unobserved link.
- **Machine-level metadata:** FP8 encoding implicit in op mode; scale value in an e-register or `vli` bits;
  granularity whole-tensor only; multiply/divide convention implied by opcode; transpose is data layout; K/M/N
  tails and padding encoded **nowhere**; verifiers check register ranges, not numerical policy.
- **Conflict found:** green card says `vli.all` clears the whole m-register file, yet atlas-mlir's attention run
  issues it with live registers and reports correct outputs (possible revision difference) → B23.

### E14. LLVM/MLIR a47bddcc build for atlas-mlir (user approved building on /scratch2)
- **Changed:** `/scratch2/loren/llvm-project` (shallow single-commit, 2.9 GB), `/scratch2/loren/tools-venv`
  (cmake 4.4.3, lit, numpy), build `/scratch2/loren/llvm-build`, install `/scratch2/loren/llvm-a47bddcc-install`.
  Lean config: Release+assertions, `mlir;lld`, `X86;RISCV`, shared libs, no tests/examples/benchmarks/docs.
  Guard script `/scratch2/loren/llvm-build-guarded.sh` stops the ninja it launched if free space < 12 GB.
- **Tested:** configure rc=0. Build in progress (5889 steps).

### E15. A4 front half — TorchAO FP8 capture through model2MLIR `90f53bc` (mixed MLP + SmolVLA)
- **Changed:** nothing tracked. m2m venv at `/scratch2/loren/model2MLIR/.venv` (CPU: torch 2.10.0, torchao 0.17.0,
  xdsl 0.65.0; **pyyaml had to be added** — `--recipe` imports merlin's `quant_recipe`). Hand-written Atlas-policy
  recipes (per-tensor e4m3) with `recipe_sha256` from `quant_recipe.digest`.
- **Tested / verified outcome** (`out/artifacts/probes/m2m_fp8_capture/20261002T060019Z/`):
  - bf16 baseline: bundle materialised, exit 0. ✅
  - dynamic FP8 without bypass: TorchAO refuses on SM 7.5 *and on CPU* ("requires CUDA compute capability ≥8.9").
  - dynamic FP8, hardware gate bypassed (**diagnostic only**), 8-wide MLP: MLIR is pure f32 — TorchAO **silently
    skips** Linears whose dims aren't multiples of 16 — yet the meta said `scheme: fp8_e4m3_dynamic_act_weight`,
    `layers_quantized: 2`. ⇒ my E8 B4 fix was a **false positive** (B4 reopened → fixed again in E17).
  - dynamic FP8, 16-wide probe: really quantized; cosine 0.9986; `_scaled_mm` left as 2 opaque `func.call`s; the
    FP8 cast lowers as an identity; source-identity relations 0/34 (LOST).
  - static PT2E FP8 with a `derive()`-shaped recipe (`quant_min/max: None`): **cosine 0.55** — TorchAO falls back to
    (0, 15) for float8, clamping all negatives to 0 (B25). With explicit ±448: cosine 0.9966.
  - Numerics: captured FP8 = TorchAO's fp32 `amax/448` scales (golden matches that reference to 1.2e-7); a po2
    reference is off by 0.10 → **captures are not Atlas-po2 numerics** (confirms B15).
  - SmolVLA (`lerobot/smolvla_base`, ~900 MB): `--dtype fp32` (keeps checkpoint's own mixed BF16) captured in
    2 m 28 s, 17,549 linalg ops, 0 opaque, 3.6 GB RSS ✅; `--dtype bf16` fails (graph asserts fp32 image input);
    needed xdsl 0.65 (0.71 breaks).
- **Metadata boundary table (FP8 static PT2E path):** FP8 payload only in `weights.safetensors` (args are f32);
  scales inline `arith.constant`; `quant_ext.quantize/dequantize_per_tensor` kept with `output_dtype=float8_e4m3fn`
  and qmin/qmax; matmul on dequantized f32; weight `linalg.transpose`; source identity complete.
  Dynamic path: weight scale is an SSA `tensor<1x1xf32>` arg; activation scale recomputed in-graph; FP8 cast LOST;
  identity LOST.

### E16. atlas-mlir built and tested against LLVM/MLIR a47bddcc
- **Changed:** `/scratch2/loren/llvm-a47bddcc-install` (594 MB) + `/scratch2/loren/llvm-build` (1.7 GB) — build took
  ~12 min, guard never tripped (min free 34 GB). `/scratch2/loren/atlas-mlir-build` (atlas-opt, atlas-emit,
  atlas-boot-pack).
- **Tested:** README chain on `test/examples/mxu.mlir`: atlas-opt ✅, atlas-emit (words `00008007 …`) ✅,
  `--convert-atlas-to-llvm` → mlir-translate → `llc -mtriple=riscv32-unknown-elf` → ELF32 RISC-V object ✅.
  Unit tests (`python -m unittest discover -s test`): standalone **177 run, 1 error, 96 skipped**; with
  `ATLAS_RTL_ROOT/ATLAS_MODEL_ROOT/ATLAS_ASSEMBLER_ROOT` = local atlas-npu: **177 run, 2 errors, 53 skipped**
  (ARC-model tests skipped: no CIRCT/ModeLIR).
- **The 2 errors:** (1) `test_handoff_examples` — this lld refuses `-Ttext=0` without `--image-base=0`; adding
  the flag links the MLP handoff ELF (entry 0x0) → upstream atlas-mlir compat bug (B24). (2)
  `test_variant_inventory` — "inspected model revision changed": local npu-model `11598ec` ≠ atlas-mlir's
  examined `5bb08624` (expected; needs the matching model revision).

### E17. Fix B4 (again) + B25 — count really-transformed layers; float8 static range
- **Changed:** `_recipe_quantizer.py`: dynamic path counts a planned layer as quantized only if its weight is no
  longer a plain `Tensor/Parameter` after `quantize_` (adds `layers_planned`; untransformed layers recorded in
  `layers_on_host` as `framework_cannot_express`). Static `_spec`: float8 with `quant_min/max` both None uses
  `torch.finfo(dtype)` (±448 e4m3fn, ±57344 e5m2).
- **Tested:** quant test set → `30 passed, 6 skipped`; `ruff check` clean. End-to-end with model2MLIR
  (`out/artifacts/probes/m2m_fp8_capture/20261002T061711Z/`, hardware gate bypassed = diagnostic):
  | case | realized_scheme | quantized/planned | cosine vs fp32 |
  |---|---|---|---|
  | b 8-wide dynamic | null (was falsely fp8) | 0/2 | 1.0 (unquantized) |
  | b2 16-wide dynamic | `fp8_e4m3_dynamic_act_weight` | 2/2 | 0.99858 |
  | c static, `derive()`-shaped recipe | `fp8_e4m3_static_act_weight` | 2 contractions | **0.99659** (was 0.55); MLIR `quant_min=-448, quant_max=448` on all 6 `quant_ext` ops |
  `test_quant_recipe.py` with `MERLIN_M2M_PYTHON`/`MERLIN_M2M_DIR` set → **15 passed, 0 skipped** (slow FP8 capture
  test actually ran).
- **Verified outcome:** ✅ B4 (v2) and B25 fixed end-to-end.

### E18. Refuse a dynamic recipe that quantizes nothing
- **Changed:** `_recipe_quantizer.py`: if every planned layer is still a plain tensor after `quantize_`, raise
  `RecipeError` naming them (previously case b exited 0, `ok: true`, with an fp32 graph under an fp8 recipe hash).
- **Tested:** `out/artifacts/probes/m2m_fp8_capture/20261002T061837Z/`: case b → exit 1, `RecipeError: quantize_
  transformed none of the 2 layers the plan assigned it (['0', '3'])`; case b2 → unchanged (2/2 quantized,
  scheme recorded; exit 3 = expected precision projection + opaque `_scaled_mm`).
- **Verified outcome:** ✅

### E19. A5 — one scaled FP8 linear executed as an Atlas program (npu-model + merlin errata)
- **Correction (user, 2026-10-02):** npu-model is a **performance model**, not a functional/numerical model. This run
  therefore shows the program executes as npu-model's ISA semantics define (program structure, data movement, layout,
  errata) — it is **not** numerical-correctness evidence. Numerical correctness needs the RTL (E9) or the RTL-exact C
  models. Merlin using npu-model as its L2 functional oracle is itself the finding (B30).
- **Changed:** nothing tracked. Program `y[32,32] = (FP8(x)·FP8(W)ᵀ)·2^(sx+sw) + b`, K=64 (2 K-tiles via
  `vmatmul.acc`), rescale = VPU multiply, bias = VPU add; assembled with atlas-npu `baremetal/assembler.py`, run
  through merlin's `functional_errata_runner` (same request shape as `program_oracle`) on npu-model `11598ec`
  (Python 3.14 venv). Headline variant `dmaconst`: 89 instructions, ~5996 model cycles.
- **atlas-mlir cross-check:** all 16 program variants translated to atlas-mlir dialect pass
  `atlas-opt --verify-atlas-machine-stream`; **`atlas-emit` words are identical to the Python assembler's for every
  program** ✅; an odd-pair `VLI.ALL 11` is rejected by the verifier (negative check ✅).
- **Verified outcome (mismatches / 1024 outputs, all errata overlays on):**
  | case | vs model-arithmetic emulation | vs SA-exact reference (RNE add) |
  |---|---|---|
  | ceil(log2(amax/448)) scales | **0** | 708 (222 ≤ 1 ulp, median 2 ulp, 4 inf) |
  | floor(log2(amax/256)) scales | **0** | 717 (58 inf) |
  | x = 0 | **0** (output == bias exactly) | **0** |
  | K = 48 zero-padded to 64 | **0** | 671 (SA C model: padded ≡ unpadded, 0/1024) |
  ⇒ the program computes exactly what npu-model's ISA semantics define; npu-model (a perf model) is **not a
  hardware-exact numerical oracle**: its `_vmatmul` computes in fp16 (+ fp16 accumulator add) → ~800/1024 accumulator
  cells differ from the RTL-exact SA model, and fp16 overflows to inf (4–58 cells) where BF16 hardware does not.
  Its `VADD` rounds RNE where RTL chops (294–316/1024 differ even with an exact MXU).
- **Errata ablation:** dropping weight-buffer lane order → output = A·W instead of A·Wᵀ (wrong) ✅ errata needed;
  dropping VMEM 4-byte unit → harmless when each DMA is followed by its VLOAD, ~959/1024 wrong when DMAs are batched;
  dropping ISA errata → `DMA.CONFIG` decodes as NOP.
- **Silent-failure finding:** without a `DMA.WAIT` after `DMA.CONFIG`, the next `DMA.LOAD` trips an IDU assertion
  that `func_program_atlas` swallows (`ignore_runtime_errors=True`) — the run reports `halted=True`, 0 unsupported
  words, and 1023/1024 wrong outputs. Found only via a strict wrapper (`strict_runner.py`).
- **Artifact:** `out/artifacts/probes/atlas_fp8_linear_model/20261002T060807Z/` (programs/*.S+.hex, mlir/, inputs,
  outputs, compare.json/txt, strict_runner.py, provenance.json).

### E20. Proof re-run (2026-10-02 ~06:30Z) — fresh executions, not citations
- **Unit tests, isolated pytest (no repo `pythonpath`):** baseline `183cbb09f` src (via `git archive`) → **9 failed,
  3 passed**; fixed working tree → **12 passed**. (A first attempt was invalid: the repo's pytest config imported the
  working-tree src for both runs.)
- **Real torchao 0.17 / torch 2.10:** `fp8_e5m2` recipe → baseline `qdata=float8_e4m3fn, act=float8_e4m3fn`;
  fixed `float8_e5m2` for both.
- **model2MLIR static FP8 capture, same recipe (`quant_min/max: None`), coverage_mlp:** baseline → MLIR
  `quant_min=0, quant_max=15`, cosine **0.5529**, argmax 0.0; fixed → `quant_min=-448, quant_max=448`, cosine
  **0.9966**, argmax 1.0.
- **VCS:** copy's RTL sources byte-identical to atlas-npu `0079c05` (sp26-fp-units `9a0cc09`, pi0-quant `8bc0f6a`,
  0 dirty RTL files); C model rebuilt from unmodified headers; regenerated vectors byte-identical to archived;
  `SystolicArrayFullMatmulTest` under VCS V-2023.12-SP1-1 (banner timestamp Oct 1 23:36 local) → PASSED; strict
  per-element compare of the fresh RTL dump → **12,288 elements, 0 bit mismatches**.
- **atlas-mlir:** 16/16 FP8-linear program variants pass `--verify-atlas-machine-stream`; `atlas-emit` words identical
  to the Python assembler's for 16/16 (first pass showed 2 "diffs" from a glob picking the wrong .hex — exact-name
  compare: identical, 89 words each).

### E21. BF16 → FP8 capture (previous FP8 captures started from FP32)
- **Context:** `coverage_mlp` is an FP32 model; earlier FP8 runs (`--dtype fp8_e4m3`) quantized FP32 → FP8.
- **Tried first:** `--dtype bf16 --recipe …` (worker casts to BF16, then applies the recipe) → **both routes fail**:
  static → `ValueError: Expected mod to be an instance of torch.nn.Module, got NoneType` (`torch.export.export(model.eval(), …)`
  in `_recipe_quantizer`; the precision-materialized module's `.eval()` returns None); dynamic → `Linear`s became
  generic `Module`s after materialization, so the layer plan refused every layer (`unmapped_module`). (B36)
- **Then:** BF16-at-source loaders (parameters + input BF16, as a BF16 checkpoint arrives) with `--dtype fp32`
  (keep the model's own precision) — `out/artifacts/probes/m2m_fp8_capture/20261002T082602Z/`:
  | route | model | scheme | agreement vs unquantized BF16 | stored weights |
  |---|---|---|---|---|
  | static PT2E | coverage_mlp (8-wide) | `fp8_e4m3_static_act_weight` (2 contractions) | cosine 0.99733, argmax 1.0 | 2 × float8_e4m3fn, 4 × bfloat16 |
  | dynamic (**diagnostic: HW gate bypassed**) | 16-wide variant | `fp8_e4m3_dynamic_act_weight` (2/2 layers) | cosine 0.99856, argmax 1.0 | 2 × float8_e4m3fn, 4 × bfloat16, 2 × float32 (scales) |
  Exit 3 on both = expected precision projection (FP8 stored as f32 in MLIR) + opaque `_scaled_mm` on dynamic.
- **Verified outcome:** ✅ BF16 → FP8 capture works when the model is BF16 at source; ❌ the worker's own
  `--dtype bf16` cast + recipe does not (B36, open).

---

### E22. Moved onto branch `track-a`, rebased onto upstream `4a7cb1821`, re-verified (2026-10-07)
- **Changed:** `origin` → `git@github.com:lorenshung/merlin.git`, `upstream` → `ucb-bar/merlin` (both already set by
  the Track B session). Fork `main` = ucb-bar `main` = `4a7cb1821` (359 commits past `183cbb09f`; `183cbb09f` has
  160 commits not upstream). Branch `track-a` created from `upstream/main`; the source patch applied cleanly to
  `_recipe_quantizer.py` and with one conflict in `_m2m_capture_worker.py` (upstream added `--integer-nonlinear` /
  `--stage-fp32` checks at the same spot) — resolved by keeping both. Upstream still has every bug fixed here.
  model2MLIR clone fast-forwarded `90f53bc` → `d671200` (the new worker imports `m2m.capture.pt2e_padding`).
- **Tested (all on the new base):**
  - Test comparison, same 13 test files + capture interpreter: pristine upstream (temporary worktree, confirmed
    pytest imported its own `src/`) → `3 failed, 68 passed`; `track-a` → `3 failed, 80 passed`. Identical failure
    set (3 × `test_quantization_spec.py`, pre-existing); +12 = the new tests. `ruff check` / `ruff format --check` clean.
  - Static BF16-source → FP8 capture, same recipe: upstream → MLIR `quant_min=0, quant_max=15`, cosine **0.4516**,
    argmax 0.0; `track-a` → `quant_min=-448, quant_max=448`, cosine **0.9973**, argmax 1.0.
  - Dynamic, 8-wide (hardware gate bypassed, diagnostic): `RecipeError: quantize_ transformed none of the 2 layers…`.
  - Dynamic, 16-wide BF16 source (diagnostic): cosine 0.9986, `fp8_e4m3_dynamic_act_weight` recorded.
- **Verified outcome:** ✅ fixes hold on current upstream.

## Commits on `track-a` (base `4a7cb1821`)

1. `fix(targetgen): pass recipe float8 dtypes and granularities to TorchAO` — B1, B2
2. `fix(targetgen): give static float8 specs the format's finite range` — B25
3. `fix(targetgen): count only layers TorchAO actually quantized` — B4, B29
4. `fix(targetgen): refuse capture dtypes with no known scheme` — B3
5. `docs(track-a): add FP8 quantization journal, bug log and PR draft`
