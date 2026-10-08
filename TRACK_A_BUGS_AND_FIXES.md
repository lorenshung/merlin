# Track A — Bugs and Fixes

One entry per defect found while working Track A (MERLIN_PLAN.md). Each entry: **where**, **symptom**
(concrete input → wrong result), **status**, **fix** (exact change), **test** (what was run and the observed
result). Status is one of: `open`, `fixed (verified)`, `fixed (not verified)`, `needs ruling` (a policy or
hardware question, not a code bug). Companion to `TRACK_A_JOURNAL.md`.

Baseline: found against merlin `183cbb09f2cd`; the code fixes were rebased onto upstream `4a7cb1821` (branch `track-a`) and re-verified there (journal E22).

| ID | Area | Summary | Status |
| --- | --- | --- | --- |
| B1 | TorchAO adapter | Dynamic FP8 config omits dtypes → E5M2 recipe silently runs as E4M3 | fixed (verified) |
| B2 | TorchAO adapter | Activation granularity taken from the weight spec | fixed (verified) |
| B3 | Capture worker | Unknown `--dtype` silently captures FP32 (the `fp8`→weight-only default is kept, see entry) | fixed (verified) |
| B4 | Capture worker / adapter | Dynamic FP8 run records `scheme: None`; first fix trusted the plan's layer count → false positive on skipped layers | fixed v2 (verified e2e) |
| B5 | Layer plan | Per-channel weight axis is `ch_axis=0` on the node feeding the contraction — after `W.t()` that is the reduction axis | open — still present on upstream `4a7cb1821` |
| B6 | Recipe ↔ TorchAO numerics | TorchAO FP8 cast keeps subnormals/exponent 15; Atlas flushes subnormals and qualifies exponents [1,14] only — mismatch not recorded | open — still present on upstream `4a7cb1821` |
| B7 | Admission | `software_allows` blocks only `unsupported`; an `unknown` admission still quantizes (fail-open) | open — still present on upstream `4a7cb1821` |
| B8 | Silent fallback | `capsule_source.derived_recipe` swallows all exceptions → falls back to named scheme | open — still present on upstream `4a7cb1821` |
| B9 | Atlas ISA model | `vpack` multiplies by scale, `vmatpop` divides with `rounding_mode="trunc"` (truncates quotient to integer) | needs ruling |
| B10 | Atlas ISA model | Reference model accumulates in fp16 and rounds to BF16 once; software spec declares per-step BF16 | needs ruling |
| B11 | Merlin errata | `functional_model_errata.yaml` corrects VPACK/VUNPACK E8M0 but not `VMATPOP_FP8_ACC_MXU{0,1}` (divide + `trunc`) | open — still present on upstream `4a7cb1821` |
| B12 | npu-model kernels | Programs use `seli eX, 1` as "unit scale"; under RTL biased-exponent semantics code 1 = 2^-126 (unit = 127) | needs ruling |
| B13 | Atlas spec | Spec models only per-step BF16 (mxu0 systolic); mxu1 inner-product tree sums a 32-tile exactly and rounds once | needs ruling |
| B14 | Atlas spec | `exponent_range: [1,14]` but RTL treats exponent-15 operands as finite normals (VCS-verified, case `e15_x_e15` bit-exact); pi0-quant's policy uses exponent 15 | needs ruling |
| B15 | TorchAO ↔ Atlas | TorchAO's public FP8 inference configs choose an fp32 `amax/448` scale; Atlas's scale register is E8M0 (power of two only). No public inference option for po2 scales (`round_scales_to_power_of_2` exists only in the training `Float8LinearConfig`) | open (design) |
| B16 | Hardware pin | Local atlas-npu checkout is clean `0079c05`; none of the 5 `local_edits` digests in `hardware_pins.yaml` match | needs ruling — pin and digests unchanged on `4a7cb1821`; local checkout still clean |
| B18 | Atlas RTL | `PushAccFP8` bias byte 0x7F/0xFF flushed to ±0 by `FPUtils.e4m3ToBf16`, but the same encoding is finite ±480 as a PE operand (RTL inconsistent; C model decodes ±480) | needs ruling |
| B19 | pi0-quant IPT model | C model intWidth uses `+15`; RTL `InnerProductTreeParams` uses `+17` (no effect on tested vectors) | needs ruling |
| B20 | Quant spec schema | `validate_quantization_declarations` accepts undeclared keys and any non-`unknown` string; no fields exist for scale rule / application point / cast numerics → no deterministic-or-refuse gate for an authored FP8 policy | open — still present on upstream `4a7cb1821` |
| B21 | TorchAO numerics | All-zero activation or weight → 100 % NaN output from torchao 0.17 per-tensor FP8 (no epsilon floor on amax) | open (recipe must define amax=0) |
| B22 | Policy | Per-tensor po2 + no FP8 subnormals: one outlier (30000) flushes 66.6 % of an activation to 0; whole-tensor rel-L2 (5.2 %) hides it (43 % on non-outlier rows) | needs ruling |
| B23 | Atlas ISA docs | Green card: `vli.all`/`vli.row` clear the whole m-register file; atlas-mlir's attention run issues `vli.all` with live m8/m12 and reports correct outputs | needs ruling |
| B24 | atlas-mlir (upstream) | `tools/export_llvm_handoff.py` links with `-Ttext=0` but no `--image-base=0`; lld 23 refuses | open (upstream) — still present on atlas-mlir `handwritten-implementation` (31 commits past 842bd46) |
| B25 | TorchAO adapter (static) | Derived float8 recipe has `quant_min/max: None` → TorchAO falls back to (0,15): negatives clamped to 0, cosine 0.55 | fixed (verified e2e) |
| B26 | model2MLIR (upstream) | `m2m.coverage.opaque_report` misses generic-form `"func.call"` → reports 0 opaque while bundle meta says 2 | fixed upstream — model2MLIR `d671200` handles generic `"func.call"` |
| B27 | model2MLIR (upstream) | Dynamic FP8 path: `aten._to_copy` to float8 lowered as identity (no rounding to the FP8 grid); FP8 payloads stored as f32 args; source identity 0/34 | open (upstream) — no FP8 types in MLIR on model2MLIR `d671200`; identity-cast part not retested |
| B28 | Env docs | Capture venv needs `pyyaml` for `--recipe`; SmolVLA workload needs xdsl 0.65 (0.71 breaks) | partly fixed upstream — model2MLIR `d671200` declares PyYAML; xdsl 0.65 half not rechecked |
| B29 | TorchAO adapter | Dynamic recipe whose planned layers TorchAO all skipped exited 0 with an fp32 graph | fixed (verified e2e) |
| B30 | Merlin oracle choice | npu-model is a **performance** model (user, 2026-10-02), yet merlin's `hardware_pins.yaml` calls it "the fast functional core the L2 tier would use" and `program_oracle` grades against it. Its `_vmatmul` (fp16 matmul + fp16 accumulate) differs from RTL-exact SA on ~800/1024 cells, with fp16 inf where BF16 HW is finite — expected for a perf model, wrong for a numerical oracle | open (merlin design) — still present on upstream `4a7cb1821` |
| B31 | npu-model (perf model) numerics | `VADD` rounds RNE; RTL chops (atlas-mlir `source-discrepancies.md:30`) → 294–316/1024 outputs differ | open (errata needed) |
| B32 | mlc runner | `func_program_atlas` runs with `ignore_runtime_errors=True`: an IDU assertion silently drops an instruction; run reports halted, 0 unsupported, 1023/1024 wrong | open (fail-open) |
| B33 | mlc runner decode | VR operand decode reads vs1 [19:13], vs2 [23:19]; RTL is [18:13], [24:19] → odd vs2 corrupts vs1 (`VMATMUL.ACC` with weight slot 1 becomes a NOP) | open |
| B34 | npu-model decode | `CSRType` registers under `cls.mnemonic` before it is set → only CSRRCI decodes; `CSRW x1, 0xC10` done-marker unsupported | open (upstream) |
| B35 | npu-model (perf model) `VLI.ALL` | Fills one register on the model, the register pair on RTL (cols 16–31 lost); odd-pair workaround rejected by atlas-mlir verifier | open (errata needed) |
| B36 | Capture worker + adapter | `--dtype bf16` with `--recipe` fails on both routes: static `torch.export.export(model.eval())` gets None from the precision-materialized module; dynamic layer plan sees generic `Module`s instead of `Linear` | open — still present on upstream `4a7cb1821` (static symptom changed: 0 contractions annotated) |
| B17 | Scale rule | pi0-quant uses `2^floor(log2(amax/256))` (amax/scale ∈ [256,512) → values in (448,512) saturate); atlas-npu `gen_mxu_vectors.py` uses `2^ceil(log2(amax/448))` | needs ruling |

---

## B1 — Dynamic FP8 config omits explicit dtypes
- **Where:** `src/merlin/targetgen/_recipe_quantizer.py` `quantize_config` (baseline lines 449-453).
- **Symptom:** recipe `weight.dtype=fp8_e5m2` → `Float8DynamicActivationFloat8WeightConfig(granularity=...)`
  with no `activation_dtype`/`weight_dtype`; TorchAO defaults both to `float8_e4m3fn`, so an E5M2 recipe is
  realised as E4M3 with no error. `activation.dtype` is never read.
- **Status:** fixed (verified).
- **Fix:** pass `activation_dtype=_torch_dtype(activation.dtype)`, `weight_dtype=_torch_dtype(weight.dtype)`;
  refuse a recipe whose activation and weight formats differ (consistent with the capture worker, which
  already raises "no capture provenance scheme for mixed recipe formats").
- **Test:** `merlin/tests/targetgen/test_recipe_quantizer_dynamic.py` (stubbed framework) — 9 passed with fix,
  8 failed on baseline. Real torchao 0.17.0 (`scratchpad/verify_b1b2.py`): baseline `fp8_e5m2` recipe →
  `qdata=float8_e4m3fn, act=float8_e4m3fn`; fixed → `float8_e5m2` for both.

## B2 — Activation granularity taken from the weight spec
- **Where:** same function (baseline line 444).
- **Symptom:** a recipe with `activation.granularity=tensor`, `weight.granularity=channel` builds
  `PerRow()` for both; the activation's own field is ignored, yet capture stats report it.
- **Status:** fixed (verified).
- **Fix:** separate maps `weight: tensor→PerTensor, channel→PerRow` and `activation: tensor→PerTensor,
  token→PerRow`; pass `granularity=(act, weight)`; refuse a mixed float8 pair by name (this TorchAO build
  scales both at one granularity); int8 refuses per-tensor activations (TorchAO's int8 dynamic is per token).
- **Test:** same file; `(tensor,tensor)→(PerTensor,PerTensor)`, `(token,channel)→(PerRow,PerRow)`,
  `(tensor act, channel w)` → `RecipeError`. Real torchao: per-tensor config forward on CPU OK.

## B3 — `fp8` capture is weight-only; unknown dtype silently FP32
- **Where:** `src/merlin/targetgen/_m2m_capture_worker.py` `_SCHEME` (44-45), `_quant_for` (192).
- **Symptom:** `--dtype fp8` / `fp8_e4m3` → `float8_weight_only_e4m3` (not the derived A+W route);
  `--dtype fp8_e5m2` (absent from `_SCHEME`) → `(None, None)` → unquantized FP32 capture, no error.
- **Status:** fixed (verified) for the unknown-dtype half. The `fp8 → float8_weight_only_e4m3` default is
  **kept**: existing whole-model recaptures (e.g. `M6_smolvla_fp8_atlas`) depend on it and the capture meta
  already records the scheme name; the derived A+W route is selected with `--recipe`.
- **Fix:** argument check — when no `--recipe`/`--scheme`/`--already-quantized` decides, a `--dtype` not in
  `_SCHEME` is an `ap.error` naming the known tokens.
- **Test:** `merlin/tests/targetgen/test_capture_dtype_is_known.py` — 3 passed; baseline worker passes
  `--dtype fp8_e5m2` through to the m2m import (would capture fp32).

## B4 — Dynamic FP8 run records `scheme: None`
- **Where:** `_m2m_capture_worker.py` `realized_scheme` (889-894) — set for FP8 only when `api == "pt2e"`.
- **Status:** fixed (not verified end-to-end — needs a working model2MLIR capture).
- **Fix:** added branch: `q.scheme` `fp8_*_dynamic_act_weight` + `api == "quantize_"` + `layers_quantized > 0`
  → `realized_scheme = q.scheme`. Key names checked against `_recipe_quantizer.py:691,695`.

## B9 / B10 — Atlas functional-model conventions (needs architect ruling)
- `examples/atlas/phase1/contracts/hwbringup_atlas_v0/isa_include/isa_definition.py:518` (vpack multiplies)
  vs `:824-828` (vmatpop divides, `torch.div(..., rounding_mode="trunc")`).
- `isa_definition.py:108-113` fp16 accumulate + single BF16 round vs `software-spec.yaml:14`
  `reduction_cadence: per_step`.
- Not a merlin code bug to fix unilaterally — the RTL decides; listed as questions for the architect.

## B11 — VMATPOP_FP8 not covered by the E8M0 errata
- **Where:** `merlin/contract/functional_model_errata.yaml` `e8m0_scale_interpretation.model_classes` lists
  only `VPACK_BF16_FP8, VUNPACK_FP8_BF16`; npu-model `VMATPOP_FP8_ACC_MXU{0,1}` computes
  `torch.div(acc, raw_byte, rounding_mode="trunc").to(float8_e4m3fn)`.
- **RTL:** `OutputConvStage.scala:85` multiplies by `2^(code-127)` (`unbExp + scale`).
- **Why not just add the class to the existing correction:** the existing wrapper returns `2^-s`, which fixes
  the direction for a divide, but `trunc` would still round the quotient to an integer, and torch's e4m3fn
  cast yields NaN on overflow where FP8Pack saturates to 448. Needs a dedicated correction kind, admitted on RTL
  evidence. **RTL evidence now exists:** VCS run (`out/artifacts/probes/atlas_mxu_vcs/20261002T054429Z/`) —
  12 scaled E4M3-pop cases (scale_exp −14..+6) bit-exact vs the pi0-quant C model, which multiplies by
  `2^(code-127)`.
- **Status:** open.

## B12 — `seli eX, 1` used as unit scale
- **Where:** npu-model `configs/programs/asm/{smolvla_requant,smolvla_fused_attention,smolvla_attention,dma_stall}.S`.
- **Symptom:** under RTL semantics `s = code - 127`, so code 1 → 2^-126: vpack saturates, vmatpop flushes.
  Unit scale is code 127. Only "correct" under the shipped (wrong) multiply-by-raw-byte functional model.
- **Status:** needs ruling (upstream npu-model; not merlin code).

## B15 — TorchAO cannot express Atlas's power-of-two scales
- **Where:** torchao 0.17.0 `Float8DynamicActivationFloat8WeightConfig` (no po2 option); `round_scales_to_power_of_2`
  appears only in `torchao/float8/config.py` (training `Float8LinearConfig`).
- **Consequence:** realising pi0-quant's policy through public TorchAO APIs needs either a merlin-side scale
  rounding step after `quantize_` or a TorchAO extension; quantified in the numerics study (in progress).
- **Status:** open (design).

## B4 (revised) — realized_scheme trusted the plan
- **Found by:** m2m capture (E15): TorchAO's float8 `quantize_` silently skips Linears whose dims are not
  multiples of 16; `layers_quantized` counted planned layers → an all-f32 capture labelled `fp8_e4m3_dynamic_act_weight`.
- **Fix v2:** `_recipe_quantizer.py` counts a layer only if its weight was replaced by a tensor subclass; the
  skipped ones are recorded with a reason. The worker's existing `layers_quantized > 0` check is now truthful.

## B25 — float8 static spec range
- **Fix:** `_spec` uses `torch.finfo(dtype)` when a float8 recipe gives no range.

## Recheck on upstream `4a7cb1821` (2026-10-07)

Every open bug a merlin or tool update could have changed was rechecked against pristine upstream `src/`
(`git archive upstream/main`), model2MLIR `d671200` and atlas-mlir `origin/handwritten-implementation`.

| Bug | Method | Result |
|---|---|---|
| B5 | Static per-channel int8 recipe from `quant_recipe.derive`, applied to `nn.Linear(8,4)` and to `x @ W.t()` with the same weight | Linear: `quantize_per_channel axis=0`, 4 scales (output channels). Transposed: `axis=0`, **8 scales** (reduction axis). Still present. |
| B6 | Searched `_recipe_quantizer.py`, `quant_recipe.py`, `_m2m_capture_worker.py` for subnormal / exponent-range recording | None. Still present. |
| B7 | Code read | `software_allows` returns `decision["status"] != "unsupported"`. Still present. |
| B8 | Code read | `derived_recipe` (`capsule_source.py:2862`) still `except Exception: return None`. Still present. |
| B11 | `functional_model_errata.yaml` | E8M0 correction still lists only `VPACK_BF16_FP8, VUNPACK_FP8_BF16`. Still present. |
| B16 | `hardware_pins.yaml` | `atlas_npu` commit `569b7c31…` and the five `local_edits` digests unchanged. |
| B20 | Atlas spec + an undeclared field and `RULING` placeholders through `validate_software_spec` | Accepted; values kept as known. Still present. |
| B30 | `hardware_pins.yaml:799` | Still "the fast functional core the L2 tier would use". |
| B36 | `--dtype bf16 --recipe` with upstream worker, static and dynamic | Both still refuse. Static now: "the model has no operator the recipe's families cover (annotated contractions=0)"; dynamic: layer plan sees generic `Module`s. |
| B24 | `tools/export_llvm_handoff.py` on atlas-mlir default branch | Still `-Ttext=0` without `--image-base`. |
| B26 | Dynamic 16-wide capture on model2MLIR `d671200` | `meta.opaque=2`, `opaque_detail={'aten__scaled_mm_default': 2}`; `m2m/coverage.py` handles the generic form. Fixed upstream. |
| B27 | Same capture | No `f8E4M3FN` types in `linalg.mlir`; `_scaled_mm` opaque. FP8 cast-as-identity not retested. |
| B28 | model2MLIR `pyproject.toml` | Declares `PyYAML>=6.0`. |

Not rechecked (unaffected by the rebase): B9, B10 (merlin's Atlas ISA model and spec unchanged upstream), B12–B19,
B21–B23 (hardware, pi0-quant and TorchAO revisions unchanged), B31–B35 (npu-model checkout unchanged; mlc not
available on this host).

