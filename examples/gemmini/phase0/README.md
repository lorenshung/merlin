# Gemmini: deterministic Phase 0

Start with [experiment.yaml](../experiment.yaml), catalog ID `gemmini-functional`.
Phase 0 has no agent: selected input bytes and pinned tools determine extraction,
requirements and candidate capsules. Unknown semantics remain explicit blockers.
Phase 1 develops the functional compiler; Phase 2 uses a separate performance cohort.

## 1. Select the inputs

| Input | Responsibility |
| --- | --- |
| [Software spec](../target/software-spec.yaml) | Operation signatures, numerical semantics, placement/transfers and quantization eligibility |
| [Hardware selection](../target/hardware.yaml) | Source-production requirements and direct RTL audit questions |
| [Host capabilities](../target/host-capabilities.yaml) | Separately pinned host compiler and reviewed operation/precision support |
| [Recipe](recipe.yaml) | Derived-only policy, comparison tolerances and oracle tiers; no authored capsule list |
| [Descriptor](../target/descriptor.yaml) | Independent iteration roster, held-out validation roster and experiment resources |
| Shared performance template | Phase 2 objectives and families, not additional functional capability |

The selected configuration has signed 8-bit operands, a 20-bit MAC result and
32-bit accumulator storage. An int32 mathematical golden needs a no-overflow
proof or an independently qualified width-aware model; storage width is not
compute precision.

[regression-seeds.yaml](regression-seeds.yaml) preserves historical authored tests
for explicit compatibility studies. It is **not** a default derivation input.
Generated capsules, weights and goldens are artifacts, never committed examples.
Read [the shared specification guide](../../../docs/guides/phase0_specification.md)
for what belongs in a SW spec versus extracted evidence.

## 2. Produce and audit one coherent RTL source selection

Follow the exact source production and audit commands in
[the target input guide](../target/README.md). The source bundle binds selected
FIRRTL, prepared generation inputs, the selected `firtool`, generated HW-MLIR and
module hierarchy. Extraction consumes that bundle explicitly; it must not silently
combine unrelated cached elaborations. Save `source-selection.json`, `facts.json`
and `validation.json` under a fresh configured artifact root.

Source consistency proves provenance, not operation legality or numerical agreement.
Manual RTL audit improves the deterministic extractor; it is not an agentic runtime
step in Phase 0. Keep raw facts separate from the effective consumer views.

## 3. Capture the independent iteration workloads

Use [the four shared loaders](../../workloads/README.md): `coverage_mlp`,
`residual_cnn`, `causal_decoder` and `multimodal_policy`. The worker seeds Python,
NumPy and Torch before loading the model and requires deterministic algorithms.
Use the same explicitly selected model2MLIR interpreter/build for all four.

```sh
"$CAPTURE_PYTHON" src/merlin/targetgen/_m2m_capture_worker.py \
  --m2m-dir "$MODEL2MLIR_ROOT" \
  --loader examples/workloads/coverage_mlp/loader.py \
  --dtype fp32 --seed 0 --materialize-bundle \
  --out "$CAPTURE_ROOT/coverage_mlp"
```

Repeat with the other three loader names and distinct output directories.
`--materialize-bundle` writes `model.mlir`, external `weights.safetensors`,
inputs/goldens and `capture_receipt.json` from the **same conversion and model
instance**; it does not recapture an unrelated model. Inspect `frontend-trace.json`
and `pytorch-opset.json` for source correspondence and build-specific operator scope.
FP32 captures inventory frontend demand; they do not imply FP32 device support.

TinyLlama, SmolVLA and ResNet50 remain held-out validation workloads. Their
captures and layer frequencies do not select or tune the derivation corpus.

## 4. Derive requirements and candidate capsules

With installed `merlin-experiments`, explicit captures and fresh facts:

Select the *same* out-of-tree support package and capability contract for
derivation and the subsequent Phase 0 run. For example, set
`MERLIN_TARGET_PATH` to the Gemmini support directory and
`MERLIN_TARGET_CONTRACT` to the reviewed selected contract before both
commands. The generated requirement binds raw facts, the contract, effective
readout facets and support-source bytes; changing a provider requires a fresh
derivation, not a resumed corpus run. The selected Gemmini readout supports
`acc_scale` but not the distinct integer-shift `requant` epilogue, so the latter
must remain an explicit rejected/host obligation rather than a fabricated
accelerator capability.

```sh
merlin experiment corpus derive gemmini-functional \
  --application-capture "coverage_mlp=$CAPTURE_ROOT/coverage_mlp/model.mlir" \
  --application-capture "residual_cnn=$CAPTURE_ROOT/residual_cnn/model.mlir" \
  --application-capture "causal_decoder=$CAPTURE_ROOT/causal_decoder/model.mlir" \
  --application-capture "multimodal_policy=$CAPTURE_ROOT/multimodal_policy/model.mlir" \
  --rtl-facts "$RTL_ROOT/facts.json" --output "$DERIVATION_ROOT"
```

Inspect `requirements.yaml`, `application-demands.json`, `synthesis-plan.json`,
`synthesis.yaml` when a candidate plan is expressible, and `derivation.json`.
The producer requires the complete declared roster and binds the selected SW spec,
recipe, workload policy and requirement bytes. A successful diagnostic derivation
is **not** a compiler certificate or reviewed corpus. Missing mappings remain
obligations; an unexpressible plan is retained as a blocked artifact.

### Realize the selected precision, then derive again

The first FP32 pass inventories source demand. Inspect the generated
`evidence/software/quantization-recipes.json` and select its matching format entry.
Resolve that entry's relative `path` against the evidence directory and set
`GENERATED_RECIPE` to the resulting JSON file. The content-addressed recipe is
generated from the selected spec and hardware; do not author a replacement.

Capture each iteration workload into a new scoped directory:

```sh
"$CAPTURE_PYTHON" src/merlin/targetgen/_m2m_capture_worker.py \
  --m2m-dir "$MODEL2MLIR_ROOT" \
  --loader examples/workloads/coverage_mlp/loader.py \
  --dtype int8 --recipe "$GENERATED_RECIPE" \
  --seed 0 --materialize-bundle \
  --out "$SCOPED_CAPTURE_ROOT/coverage_mlp"
```

Repeat for the other three loaders. The recipe scopes eligible contractions;
normalization, embeddings and unsupported operations are not blanket-quantized.
Inspect each `meta.json` for `quantization_stats`, calibration count/source,
`recipe_agreement` and `integerization_receipt`. Integer realization, agreement
with the portable quantized graph, and error against the original FP32 model
are separate observations. A single synthetic calibration example is a smoke
input, not workload-accuracy validation or proof of accelerator execution.

Derive a fresh corpus plan from these exact realized bundles:

```sh
merlin experiment corpus derive gemmini-functional \
  --application-capture "coverage_mlp=$SCOPED_CAPTURE_ROOT/coverage_mlp/model.mlir" \
  --application-capture "residual_cnn=$SCOPED_CAPTURE_ROOT/residual_cnn/model.mlir" \
  --application-capture "causal_decoder=$SCOPED_CAPTURE_ROOT/causal_decoder/model.mlir" \
  --application-capture "multimodal_policy=$SCOPED_CAPTURE_ROOT/multimodal_policy/model.mlir" \
  --rtl-facts "$RTL_ROOT/facts.json" --output "$REALIZED_DERIVATION_ROOT"
```

Preserve the bootstrap plan and FP32 bundles. For the next step, select
`REALIZED_DERIVATION_ROOT`, not the initial FP32 derivation. New recipe, framework,
source or calibration bytes require newly captured bundles and a fresh plan.
Derivation checks each quantized capture's recorded recipe digest against the
recipes derived from the *currently selected* provider and SW spec; captures
from an older provider cannot be reused just because their MLIR parses.

`evidence/evidence-manifest.json` links to the exact input bytes and consumer views:

- `hardware/circt/facts.json`: byte-identical extraction.
- `hardware/effective-views/`: actual profile, readout, quantization and execution inputs.
- `software/source-snapshots/`: selected RTL/source/parser inputs with hashes.
- `software/frontend/` and `software/framework/`: saved lineage and operator catalogs.
- `coverage/operation-accounting.json`: per-model and combined accelerator/host/unresolved splits, signatures and precisions.
- `software/quantization-contract.json`: format alternatives, eligibility and unknowns.
- `software/quantization-recipes.json`: generated prospective recipes and their exact byte identities.
- `coverage/README.md`: automatically rendered navigation of those same reports.

## 5. Generate a fresh corpus and keep cohorts distinct

Select the realized requirement/profile together for inspect, preflight and run:
Pin the same Model2MLIR checkout and capture interpreter used to realize the
iteration workloads. In particular, a sibling checkout may lack the selected
recipe's integerization implementation; Phase 0 must fail instead of falling
back to a different capture path.

```sh
MERLIN_MODEL2MLIR="$MODEL2MLIR_ROOT" MERLIN_M2M_PYTHON="$CAPTURE_PYTHON" \
  merlin experiment run gemmini-functional --phase 0 \
  --phase0-rtl-facts "$RTL_ROOT/facts.json" \
  --phase0-conformance-spec "$REALIZED_DERIVATION_ROOT/requirements.yaml" \
  --phase0-synth-profile "$REALIZED_DERIVATION_ROOT/synthesis.yaml" \
  --phase0-evidence-mode diagnostic --run-dir "$RUN_ROOT"
```

Inspect `<run>/phase0/coverage/generation.json` for written, omitted and failed
members, then `capsules/MANIFEST.yaml` for distinct functional, performance and
diagnostic selections. A candidate list is not proof that its writers/oracles can
execute every member. Provision the selected independent numerical model and
OOT toolchain before execution; missing mandatory private evidence blocks admission.

The separate `phase1-capsule-coverage.json` and `phase2-capsule-coverage.json`
reports inventory only each selected cohort's exact bytes. Interface-command
observations and source-model MLIR witnesses remain distinct; a performance
cohort cannot borrow functional source coverage or claim whole-model validation.

### Review and release

A successful controller receipt proves that the selected generator finished; it
does not establish capsule coverage or target execution. Check
`coverage/generation.json` for zero writer failures **and** zero omissions, then
inspect `phase1-capsule-coverage.json` and `phase2-capsule-coverage.json`
separately. The derived micro-model composition test reads the run's frozen
iteration captures, not an ambient `out/artifacts/recaptures` directory.
The memory-mapping axis reads the exact CIRCT facts bytes recorded in
`capsules/_phase0/coverage-inputs.json`; a missing or changed facts snapshot
leaves that axis unmeasured. This is pre-compiler corpus coverage, not evidence
that a compiled program used the on-chip store correctly.

Verified admission also requires reviewed software and host semantics, complete
capture source closure, independent numerical and compiler checks, resolved
operation/transfer obligations, and an operator-owned hidden cohort. These
cannot be inferred from a diagnostic run or supplied by changing a status
field. Review the generated model inventory and `grading.resource_bound` for
the *selected* cohort; if policy changes, freeze a new run rather than editing
an old receipt. With a separately selected private hidden category, prepare a
candidate release using:

```sh
merlin experiment corpus prepare "$NEW_RUN_ROOT" --generated-only \
  --private-baseline "$PRIVATE_HIDDEN_ROOT" --output "$NEW_RELEASE_ROOT"
```

Preparation reports unmatched public models and a missing or withheld hidden
cohort; it does not choose exclusions for the operator. Acknowledging a seal
does not repair incomplete coverage or missing execution receipts.

Review coverage, placement and independent numerical checks before preparing
[the reviewed Phase 0 handoff](../../../experiments/README.md#reviewed-phase-0-handoff).
Changing a status field cannot qualify old artifacts. New inputs require newly
frozen runs; preserve old outputs unchanged. See [the artifact map](../artifacts/README.md)
and [whole-model walkthrough](../whole-model/README.md) for member MLIR, external
tensors and intermediate lowering snapshots.
