# Headline model capture and lowering

TinyLlama, SmolVLA and ResNet50 are held-out validation workloads, not inputs to
Phase 0 capsule selection. Their real model loaders live in model2MLIR. Merlin
does not copy those implementations or infer target support from host lowering.
Use a fresh artifact directory for every run.

| Model | Capture scope | What one capture does **not** prove |
| --- | --- | --- |
| TinyLlama | Full checkpoint, one prefill forward (`M2M_SEQ=8`) | KV-cache decode/session correctness, real-corpus quality, accelerator support |
| SmolVLA | Full checkpoint, explicit prefix → recurrent denoise → action-decode programs | Policy quality on a real trajectory, target execution |
| ResNet50 | Full pretrained `IMAGENET1K_V2` checkpoint, one synthetic image | ImageNet accuracy, real-input preprocessing, target execution |

The commands below are compiler checks with seeded synthetic inputs. For
application-quality validation, supply attributed real inputs through each
model2MLIR loader's documented environment and inspect its `paper_ready` and
source fields. Do not turn a truncated, random-initialized, or single-step
diagnostic into a whole-model claim.

## Capture from PyTorch

Choose the model2MLIR checkout and its model-specific interpreter, then run the
Merlin worker from this repository root. Keep checkpoints cached and use offline
mode when checking an existing local selection. The TinyLlama adapter
[tiny_llama_loader.py](tiny_llama_loader.py) delegates model construction to
model2MLIR; it only records the ordinary prefill path's otherwise missing
checkpoint and input scope. It verifies that cached config and weights resolve
to the same revision and that the loaded layer count matches that config.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 M2M_SEQ=8 \
  "$LLAMA_CAPTURE_PYTHON" src/merlin/targetgen/_m2m_capture_worker.py \
  --m2m-dir "$MODEL2MLIR_ROOT" \
  --loader examples/workloads/headline_validation/tiny_llama_loader.py \
  --dtype fp32 --seed 0 --materialize-bundle \
  --out "$HEADLINE_ROOT/tinyllama-prefill"

M2M_RESNET_RANDOM=1 M2M_SESSION_STEPS=1 \
  "$RESNET_CAPTURE_PYTHON" src/merlin/targetgen/_m2m_capture_worker.py \
  --m2m-dir "$MODEL2MLIR_ROOT" \
  --loader "$MODEL2MLIR_ROOT/workloads/resnet50_v1_5/loader.py" \
  --dtype fp32 --seed 0 --materialize-bundle \
  --out "$HEADLINE_ROOT/resnet50-synthetic"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  M2M_SMOLVLA_PRETRAINED=1 M2M_SMOLVLA_SESSION=e2e \
  "$SMOLVLA_CAPTURE_PYTHON" src/merlin/targetgen/_m2m_capture_worker.py \
  --m2m-dir "$MODEL2MLIR_ROOT" \
  --loader "$MODEL2MLIR_ROOT/workloads/smolvla/loader.py" \
  --dtype fp32 --seed 0 --materialize-bundle \
  --out "$HEADLINE_ROOT/smolvla-session"
```

Do not set `M2M_RESNET_PRETRAINED=0` for a full-checkpoint check. ResNet's
`M2M_RESNET_RANDOM=1` above selects synthetic *inputs*, not random weights.
SmolVLA's `--dtype fp32` selects an unquantized session, not an all-f32 tensor
ABI: the captured KV cache has bf16 leaves. Inspect `meta.json` per stage.

Before lowering, require `ok: true`, `opaque: 0`, the intended checkpoint and
scope, and `capture_receipt.json` in every selected bundle. The SmolVLA root
`session-receipt.json` and `session_contract.yaml` must name all three programs
and the prefix/cache/flow bindings. The receipt's
`source_closure_verified: false` is a blocking fact for verified release, not a field to edit. A capture
may still be useful for diagnostic compiler checks. Inspect each stage's
`frontend-trace.json` separately: `ok: true` and zero opaque calls do not imply
complete PyTorch-to-MLIR operation correspondence. A `diagnostic` trace leaves
that lineage obligation open even if later LLVM lowering succeeds.

## Lower and inspect every program

`merlin lower` consumes one complete `model.mlir` at a time. Run it with the
selected compiler interpreter and MLIR toolchain configured as in
[the lowering guide](../../../docs/guides/model_lowering.md). For TinyLlama and
ResNet50, `CAPTURE` is their bundle. For SmolVLA, repeat once each for
`stages/prefix_encode`, `stages/flow_denoise`, and `stages/action_decode`:

```sh
merlin lower "$CAPTURE/model.mlir" --out "$LOWER_ROOT/$PROGRAM" \
  --ir-audit exact \
  --audit-sidecar "$CAPTURE/weights.safetensors" \
  --audit-sidecar "$CAPTURE/weights.safetensors.manifest.json"
```

`model.mlir` retains typed operation and provenance attributes; its weights
and biases are in the parallel safetensors file. The lowering result's
`audit_index` points to named intermediate MLIR and LLVM IR stages under the
fresh output directory. Inspect the index and terminal `model.ll` for **each**
program. Successful host LLVM lowering proves neither OOT accelerator codegen
nor numerical execution. Phase 1 must still account for host/accelerator
placement, precision, complete sessions and independent numerical results.
